import re
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from newsfeed import (
    Article,
    ExclusionRules,
    candidate_clusters,
    canonicalize_url,
    exact_deduplicate,
    is_excluded_article,
    render_feed,
    review_with_gemini,
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
            title_patterns=(re.compile(r"\b(sport\w*|voetbal\w*)\b", re.IGNORECASE),),
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

    def test_canonical_url_drops_tracking(self) -> None:
        self.assertEqual(
            canonicalize_url("https://Example.com/news/?utm_source=x&id=2#top"),
            "https://example.com/news?id=2",
        )

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
            article("a", "Gebeurtenis met alle feiten", "https://a.example/1", "Feiten A en B"),
            article("b", "Gebeurtenis kort gemeld", "https://b.example/2", "Feit A"),
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
    def test_invalid_gemini_decision_keeps_every_candidate(self, post: Mock) -> None:
        items = [
            article("a", "Gebeurtenis met feiten", "https://a.example/1", "Feiten A"),
            article("b", "Gebeurtenis gemeld", "https://b.example/2", "Feit A"),
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

    def test_rendered_feed_is_valid_xml(self) -> None:
        item = article("a", "Titel & nuance", "https://example.com/a", "Samenvatting <veilig>")
        feed = render_feed([item], NOW, "https://example.com/news")
        root = ET.fromstring(feed)
        self.assertEqual(root.tag, "rss")
        self.assertEqual(root.findtext("./channel/item/title"), "Titel & nuance")


if __name__ == "__main__":
    unittest.main()
