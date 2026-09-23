import re
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from newsfeed import (
    Article,
    ExclusionRules,
    build_source_cutoffs,
    candidate_clusters,
    canonicalize_url,
    decode_embedding,
    encode_embedding,
    exact_deduplicate,
    fetch_embedding_batch,
    fetch_metadata_description,
    gemini_payload,
    get_embeddings,
    hostname_is_public,
    is_excluded_article,
    parse_beehiiv_archive_articles,
    render_feed,
    review_with_gemini,
    safe_metadata_url,
)

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def article(article_id: str, title: str, link: str, summary: str, source: str = "Test") -> Article:
    return Article(
        article_id=article_id,
        title=title,
        link=link,
        summary=summary,
        source=source,
        source_url="https://example.com/feed",
        published=NOW,
        fetched_at=NOW,
    )


class NewsfeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sport_rules = ExclusionRules(
            url_path_segments=frozenset({"sport", "voetbal"}),
            category_terms=frozenset({"sport", "voetbal"}),
            title_patterns=(re.compile(r"\b(sport\w*|hengelsport\w*|voetbal\w*)\b", re.IGNORECASE),),
        )

    def test_sport_url_is_excluded(self) -> None:
        self.assertTrue(
            is_excluded_article(
                "Club presenteert nieuwe trainer",
                "https://example.com/voetbal/nieuws",
                (),
                self.sport_rules,
            )
        )

    def test_gossip_url_path_is_excluded_without_title_filtering(self) -> None:
        rules = ExclusionRules(
            url_path_segments=frozenset({"show", "entertainment", "achterklap"}),
            category_terms=frozenset(),
            title_patterns=(),
        )
        self.assertTrue(
            is_excluded_article(
                "Bekende Nederlander geeft interview",
                "https://www.ad.nl/show/interview~a123/",
                (),
                rules,
            )
        )
        self.assertFalse(
            is_excluded_article(
                "Onderzoek toont nieuwe resultaten",
                "https://example.com/wetenschap/showcase-onderzoek",
                (),
                rules,
            )
        )

    def test_sport_category_is_excluded(self) -> None:
        self.assertTrue(
            is_excluded_article(
                "Nieuwe trainer gepresenteerd",
                "https://example.com/nieuws/1",
                ("Voetbal",),
                self.sport_rules,
            )
        )

    def test_sport_title_is_excluded_but_transport_is_not(self) -> None:
        self.assertTrue(
            is_excluded_article(
                "Voetbalteam wint finale",
                "https://example.com/nieuws/1",
                (),
                self.sport_rules,
            )
        )
        self.assertFalse(
            is_excluded_article(
                "Kabinet investeert in openbaar transport",
                "https://example.com/economie/2",
                (),
                self.sport_rules,
            )
        )
        self.assertTrue(
            is_excluded_article(
                "Hengelsport haalt opgelucht adem",
                "https://example.com/binnenland/3",
                (),
                self.sport_rules,
            )
        )

    def test_canonical_url_drops_tracking(self) -> None:
        self.assertEqual(
            canonicalize_url("https://Example.com/news/?utm_source=x&id=2#top"),
            "https://example.com/news?id=2",
        )
        self.assertEqual(canonicalize_url("javascript:alert(1)"), "")

    def test_weekly_source_can_use_a_longer_history_window(self) -> None:
        cutoffs, overrides = build_source_cutoffs(
            [
                {"name": "Dagelijks"},
                {"name": "Wekelijks", "history_hours": 240},
            ],
            NOW,
            72,
        )
        self.assertEqual(cutoffs["Dagelijks"], NOW - timedelta(hours=72))
        self.assertEqual(cutoffs["Wekelijks"], NOW - timedelta(hours=240))
        self.assertEqual(overrides, {"Wekelijks": 240})

    def test_beehiiv_archive_uses_only_public_card_metadata(self) -> None:
        raw = b"""
        <html><body>
          <a href="/p/nieuw-bericht">
            <div>
              <h2>Nieuw AI-bericht met nuance</h2>
              <p>Openbare korte uitleg uit het archief.</p>
              <time datetime="2026-09-21T08:30:00Z">Sep 21, 2026</time>
              <span>12 min read</span>
            </div>
          </a>
        </body></html>
        """
        source = {
            "name": "AI Report",
            "url": "https://www.aireport.nl/archive",
            "article_domains": ["aireport.nl"],
        }
        articles, fetched, excluded = parse_beehiiv_archive_articles(
            raw,
            source,
            NOW,
            NOW - timedelta(hours=240),
            self.sport_rules,
        )
        self.assertEqual(fetched, 1)
        self.assertEqual(excluded, 0)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].title, "Nieuw AI-bericht met nuance")
        self.assertEqual(
            articles[0].summary, "Openbare korte uitleg uit het archief."
        )
        self.assertEqual(
            articles[0].link, "https://www.aireport.nl/p/nieuw-bericht"
        )
        self.assertEqual(
            articles[0].published,
            datetime(2026, 9, 21, 8, 30, tzinfo=timezone.utc),
        )

    @patch("newsfeed.hostname_is_public", return_value=True)
    def test_metadata_url_requires_https_and_allowlisted_domain(self, _public: Mock) -> None:
        self.assertTrue(safe_metadata_url("https://www.trouw.nl/a", ("trouw.nl",)))
        self.assertFalse(safe_metadata_url("http://www.trouw.nl/a", ("trouw.nl",)))
        self.assertFalse(safe_metadata_url("https://example.com/a", ("trouw.nl",)))
        self.assertFalse(safe_metadata_url("https://user:pass@www.trouw.nl/a", ("trouw.nl",)))

    @patch("newsfeed.socket.getaddrinfo")
    def test_private_metadata_target_is_rejected(self, getaddrinfo: Mock) -> None:
        getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        self.assertFalse(hostname_is_public("www.trouw.nl"))

    @patch("newsfeed.hostname_is_public", return_value=True)
    def test_fetches_only_metadata_description(self, _public: Mock) -> None:
        response = Mock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.encoding = "utf-8"
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [
            b'<html><head><meta property="og:description" content="Publieke samenvatting">',
            b'</head><body>Betaalde artikeltekst die niet verder gelezen hoeft te worden</body>',
        ]
        session = Mock()
        session.get.return_value = response
        description = fetch_metadata_description(
            session, "https://www.trouw.nl/a", ("trouw.nl",)
        )
        self.assertEqual(description, "Publieke samenvatting")
        self.assertEqual(session.get.call_args.kwargs["allow_redirects"], False)
        self.assertLessEqual(sum(map(len, response.iter_content.return_value)), 131_072)

    def test_exact_deduplication_removes_tracking_variant(self) -> None:
        first = article("a", "Titel", "https://example.com/a?utm_source=x", "Samenvatting")
        second = article("b", "Andere kop", "https://example.com/a", "Andere samenvatting")
        unique, removed = exact_deduplicate([first, second])
        self.assertEqual(removed, 1)
        self.assertEqual(len(unique), 1)

    def test_candidate_cluster_finds_related_event(self) -> None:
        items = [
            article(
                "a",
                "Kabinet presenteert nieuw klimaatplan voor industrie",
                "https://a.example/1",
                "Nieuwe klimaatregels voor de Nederlandse industrie zijn bekendgemaakt.",
                "Bron A",
            ),
            article(
                "b",
                "Nieuw klimaatplan kabinet raakt zware industrie",
                "https://b.example/2",
                "Het klimaatplan bevat regels en subsidies voor zware industrie.",
                "Bron B",
            ),
            article("c", "Voetbalclub wint bekerfinale", "https://c.example/3", "De finale eindigde in 2-1."),
        ]
        clusters = candidate_clusters(items, 36)
        self.assertEqual(len(clusters), 1)
        self.assertEqual({item.article_id for item in clusters[0]}, {"a", "b"})

    def test_embeddings_can_find_event_without_word_overlap(self) -> None:
        items = [
            article("a", "AI-platform getroffen door aanvallers", "https://a.example/1", "Beveiligingsincident"),
            article("b", "Waarom de Hugging Face-hack meevalt", "https://b.example/2", "Technische analyse"),
            article("c", "Kabinet presenteert begroting", "https://c.example/3", "Nieuwe plannen"),
        ]
        embeddings = {
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.99, 0.05, 0.0], dtype=np.float32),
            "c": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        clusters = candidate_clusters(items, 36, embeddings, 0.95)
        self.assertEqual(len(clusters), 1)
        self.assertEqual({item.article_id for item in clusters[0]}, {"a", "b"})

    def test_embedding_cache_roundtrip_is_close(self) -> None:
        vector = np.array([-0.4, 0.0, 0.25, 0.8], dtype=np.float32)
        decoded = decode_embedding(encode_embedding(vector), 4)
        np.testing.assert_allclose(decoded, vector, atol=0.007)

    def test_without_api_key_embeddings_fall_back_to_jaccard(self) -> None:
        with TemporaryDirectory() as directory:
            embeddings, status = get_embeddings(
                [article("a", "Titel", "https://a.example/1", "Samenvatting")],
                None,
                "gemini-embedding-2",
                768,
                50,
                40,
                cache_path=Path(directory) / "embeddings.json",
            )
        self.assertIsNone(embeddings)
        self.assertEqual(status["state"], "not_configured_jaccard")

    @patch("newsfeed.fetch_embedding_batch", side_effect=RuntimeError("quota bereikt"))
    def test_embedding_failure_is_not_sent_again(self, fetch: Mock) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            embeddings, status = get_embeddings(
                [article("a", "Titel", "https://a.example/1", "Samenvatting")],
                "secret",
                "gemini-embedding-2",
                3,
                50,
                40,
                cache_path=cache_path,
            )
            embeddings_again, second_status = get_embeddings(
                [article("a", "Titel", "https://a.example/1", "Samenvatting")],
                "secret",
                "gemini-embedding-2",
                3,
                50,
                40,
                cache_path=cache_path,
            )
            self.assertTrue(cache_path.exists())
        self.assertIsNone(embeddings)
        self.assertIsNone(embeddings_again)
        self.assertEqual(status["state"], "failed_jaccard")
        self.assertEqual(second_status["state"], "one_shot_jaccard")
        self.assertEqual(second_status["skipped_previously_attempted"], 1)
        self.assertEqual(fetch.call_count, 1)

    @patch("newsfeed.fetch_embedding_batch")
    def test_successful_embedding_is_sent_once(self, fetch: Mock) -> None:
        fetch.return_value = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
        item = article("a", "Titel", "https://a.example/1", "Samenvatting")
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "embeddings.json"
            first, _ = get_embeddings(
                [item], "secret", "gemini-embedding-2", 3, 50, 40, cache_path
            )
            second, status = get_embeddings(
                [item], "secret", "gemini-embedding-2", 3, 50, 40, cache_path
            )
        self.assertEqual(set(first or {}), {"a"})
        self.assertEqual(set(second or {}), {"a"})
        self.assertEqual(status["cached"], 1)
        self.assertEqual(fetch.call_count, 1)

    @patch("newsfeed.fetch_embedding_batch")
    def test_successful_embedding_batch_is_kept_if_next_batch_fails(self, fetch: Mock) -> None:
        fetch.side_effect = [
            [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
            RuntimeError("quota bereikt"),
        ]
        items = [
            article("a", "Titel A", "https://a.example/1", "Lange samenvatting A"),
            article("b", "Titel B", "https://b.example/2", "Lange samenvatting B"),
        ]
        with TemporaryDirectory() as directory:
            embeddings, status = get_embeddings(
                items,
                "secret",
                "gemini-embedding-2",
                3,
                1,
                2,
                cache_path=Path(directory) / "embeddings.json",
            )
        self.assertEqual(set(embeddings or {}), {"a"})
        self.assertEqual(status["state"], "failed_partial_jaccard")
        self.assertEqual(status["available"], 1)

    @patch("newsfeed.requests.post")
    def test_embedding_batch_uses_header_and_clustering_prefix(self, post: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}
        post.return_value = response
        vectors = fetch_embedding_batch(
            [article("a", "Titel", "https://a.example/1", "Samenvatting")],
            "secret",
            "gemini-embedding-2",
            3,
        )
        self.assertEqual(vectors[0].shape, (3,))
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "secret")
        self.assertNotIn("params", kwargs)
        request = kwargs["json"]["requests"][0]
        self.assertEqual(request["embedContentConfig"]["outputDimensionality"], 3)
        self.assertTrue(request["content"]["parts"][0]["text"].startswith("task: clustering | query:"))

    def test_gemini_payload_uses_low_thinking_without_temperature(self) -> None:
        item = article("a", "Titel", "https://a.example/1", "Samenvatting")
        generation_config = gemini_payload([[item]])["generationConfig"]
        self.assertNotIn("temperature", generation_config)
        self.assertEqual(generation_config["thinkingConfig"]["thinkingLevel"], "LOW")

    def test_without_api_key_candidates_are_kept(self) -> None:
        items = [
            article("a", "Zelfde gebeurtenis met nieuwe feiten", "https://a.example/1", "Feiten A"),
            article("b", "Zelfde gebeurtenis en gevolgen", "https://b.example/2", "Gevolgen B"),
        ]
        reviewed, status = review_with_gemini(items, [items], None, "gemini-test")
        self.assertEqual(reviewed, items)
        self.assertEqual(status["state"], "not_configured_exact_only")

    @patch("newsfeed.requests.post")
    def test_valid_gemini_decision_removes_only_redundant_item(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis met alle feiten",
                "https://a.example/1",
                "Deze samenvatting bevat alle relevante feiten A en B voor de beoordeling.",
            ),
            article(
                "b",
                "Gebeurtenis kort gemeld",
                "https://b.example/2",
                "Deze samenvatting bevat alleen feit A en voegt verder niets nieuws toe.",
            ),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"remove":[{"id":"b","duplicate_of":"a","reason":"redundant"}]}'
                            }
                        ]
                    }
                }
            ]
        }
        post.return_value = response
        reviewed, status = review_with_gemini(items, [items], "secret", "gemini-test")
        self.assertEqual([item.article_id for item in reviewed], ["a"])
        self.assertEqual(status["state"], "ok")
        self.assertNotIn("params", post.call_args.kwargs)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "secret")

    @patch("newsfeed.requests.post")
    def test_gemini_review_is_reused_without_resending_articles(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis met alle feiten",
                "https://a.example/1",
                "Deze samenvatting bevat alle relevante feiten voor een goede vergelijking.",
            ),
            article(
                "b",
                "Gebeurtenis redundant gemeld",
                "https://b.example/2",
                "Deze samenvatting herhaalt dezelfde feiten en voegt inhoudelijk niets toe.",
            ),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"remove":[{"id":"b","duplicate_of":"a","reason":"redundant"}]}'
                            }
                        ]
                    }
                }
            ]
        }
        post.return_value = response
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "reviews.json"
            first, _ = review_with_gemini(
                items, [items], "secret", "gemini-test", cache_path
            )
            # Feeds without a real timestamp may receive a moving fallback date;
            # that alone must not cause the article text to be sent again.
            for item in items:
                item.published += timedelta(minutes=30)
            second, status = review_with_gemini(
                items, [items], "secret", "gemini-test", cache_path
            )
        self.assertEqual([item.article_id for item in first], ["a"])
        self.assertEqual([item.article_id for item in second], ["a"])
        self.assertEqual(status["removed_from_cache"], 1)
        self.assertEqual(post.call_count, 1)

    @patch("newsfeed.requests.post", side_effect=RuntimeError("tijdelijke fout"))
    def test_failed_gemini_review_is_not_sent_again(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis uitgebreid gemeld",
                "https://a.example/1",
                "Deze samenvatting bevat voldoende feiten voor een inhoudelijke vergelijking.",
            ),
            article(
                "b",
                "Gebeurtenis ook gemeld",
                "https://b.example/2",
                "Deze samenvatting bevat eveneens voldoende feiten voor de vergelijking.",
            ),
        ]
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "reviews.json"
            first, first_status = review_with_gemini(
                items, [items], "secret", "gemini-test", cache_path
            )
            second, second_status = review_with_gemini(
                items, [items], "secret", "gemini-test", cache_path
            )
        self.assertEqual(first, items)
        self.assertEqual(second, items)
        self.assertEqual(first_status["state"], "failed_exact_only")
        self.assertEqual(second_status["state"], "no_reviewable_candidates")
        self.assertEqual(post.call_count, 1)

    @patch("newsfeed.requests.post")
    def test_invalid_gemini_decision_keeps_every_candidate(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis met feiten",
                "https://a.example/1",
                "Deze samenvatting bevat genoeg feiten voor een geldige vergelijking.",
            ),
            article(
                "b",
                "Gebeurtenis gemeld",
                "https://b.example/2",
                "Deze samenvatting is ook lang genoeg voor een geldige vergelijking.",
            ),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": '{"remove":[{"id":"onbekend","duplicate_of":"a","reason":"x"}]}'}]}}
            ]
        }
        post.return_value = response
        reviewed, status = review_with_gemini(items, [items], "secret", "gemini-test")
        self.assertEqual(reviewed, items)
        self.assertEqual(status["state"], "failed_exact_only")

    @patch("newsfeed.requests.post")
    def test_gemini_cannot_remove_title_only_item(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis uitgebreid gemeld",
                "https://a.example/1",
                "Deze samenvatting bevat voldoende feiten voor een inhoudelijke vergelijking.",
            ),
            article("b", "Gebeurtenis kort gemeld", "https://b.example/2", "Kort."),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"remove":[{"id":"b","duplicate_of":"a","reason":"redundant"}]}'
                            }
                        ]
                    }
                }
            ]
        }
        post.return_value = response
        reviewed, status = review_with_gemini(items, [items], "secret", "gemini-test")
        self.assertEqual(reviewed, items)
        self.assertEqual(status["state"], "no_reviewable_candidates")
        post.assert_not_called()

    @patch("newsfeed.requests.post")
    def test_short_item_is_kept_while_reviewable_pair_is_processed(self, post: Mock) -> None:
        items = [
            article(
                "a",
                "Gebeurtenis met alle feiten",
                "https://a.example/1",
                "Deze samenvatting bevat alle relevante feiten voor de vergelijking.",
            ),
            article(
                "b",
                "Gebeurtenis redundant gemeld",
                "https://b.example/2",
                "Deze samenvatting herhaalt dezelfde relevante feiten zonder toevoeging.",
            ),
            article("c", "Gebeurtenis zonder uitleg", "https://c.example/3", "Kort."),
        ]
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"remove":[{"id":"b","duplicate_of":"a","reason":"redundant"}]}'
                            }
                        ]
                    }
                }
            ]
        }
        post.return_value = response
        reviewed, status = review_with_gemini(items, [items], "secret", "gemini-test")
        self.assertEqual([item.article_id for item in reviewed], ["a", "c"])
        self.assertEqual(status["state"], "ok")
        self.assertEqual(status["articles_skipped_insufficient_summary"], 1)

    def test_rendered_feed_is_valid_xml(self) -> None:
        item = article("a", "Titel & nuance", "https://example.com/a", "Samenvatting <veilig>")
        feed = render_feed([item], NOW, "https://example.com/news")
        root = ET.fromstring(feed)
        self.assertEqual(root.tag, "rss")
        self.assertEqual(root.findtext("./channel/item/title"), "Titel & nuance")


if __name__ == "__main__":
    unittest.main()
