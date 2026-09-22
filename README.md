# Persoonlijke AI-nieuwsfeed

Deze repository combineert tien RSS-bronnen tot één feed voor Feedly. De pipeline
verwijdert eerst exacte dubbelen, maakt daarna lokaal goedkope kandidaatclusters
en stuurt uitsluitend die clusters naar Gemini. Artikelen over dezelfde gebeurtenis
blijven staan wanneer ze aantoonbaar extra informatiewaarde hebben.

De gepubliceerde feed komt na het activeren van GitHub Pages beschikbaar op:

`https://miliaan82.github.io/nieuwsfeed/feed.xml`

## Werking

1. GitHub Actions start elke 30 minuten.
2. `newsfeed.py` haalt de RSS-feeds op.
3. Sportartikelen worden verwijderd op basis van de RSS-categorie, het URL-pad en
   een beperkte titelcontrole. De regels staan in `config/sources.json`.
4. De aggregator voegt de nieuwe items samen met maximaal 72 uur historie uit een
   private GitHub Actions-cache.
5. Exact gelijke URL's (zonder trackingparameters) en exact gelijke RSS-inhoud
   worden lokaal verwijderd.
6. Alleen wanneer een RSS-samenvatting ontbreekt of erg kort is, leest de bot een
   begrensd deel van de publieke artikelpagina om uitsluitend de OpenGraph- of
   meta-description op te halen. De artikeltekst wordt niet verwerkt.
7. Gemini Embedding 2 maakt van de overgebleven RSS-metadata semantische
   kandidaatclusters. Eerder berekende vectors worden privé gecachet; bij een
   API-fout neemt de lokale tekstvergelijking het automatisch over.
8. Alleen de titel, RSS-samenvatting/paginametadata en RSS-metadata van die
   clusters gaan naar Gemini Flash voor de uiteindelijke behoudsbeslissing.
9. De workflow publiceert `public/feed.xml`, `public/index.html` en
   `public/status.json` rechtstreeks via GitHub Pages.

Gemini werkt conservatief: dezelfde gebeurtenis is niet genoeg om iets te
verwijderen. Nieuwe feiten, primaire bronnen, expertise, gevolgen, onzekerheid,
correcties, nuances en wezenlijk andere interpretaties blijven behouden. Andere
toon of framing zonder extra informatie is geen zelfstandig behoudsargument.

### Fail-safe

Als de API-sleutel ontbreekt, Gemini niet bereikbaar is, het gratis quotum op is
of de respons ongeldig is,
wordt geen enkel mogelijk inhoudelijk duplicaat verwijderd. Alleen de voorafgaande
exacte deduplicatie blijft dan actief. Een bronstoring blokkeert de overige bronnen
niet; nog geldige items uit de vorige feed blijven maximaal 72 uur beschikbaar.

## Eenmalige configuratie op GitHub

1. Maak in Google AI Studio een apart Google-project zonder gekoppelde
   betaalmethode. Activeer geen betaalde tier.
2. Open **Settings → Secrets and variables → Actions → New repository secret**.
3. Maak het secret `GEMINI_API_KEY` met de sleutel uit dat gratis project.
4. Open **Settings → Pages** en kies bij **Source** voor **GitHub Actions**.
5. Open **Actions → Update news feed → Run workflow** voor de eerste handmatige run.
6. Voeg daarna de bovenstaande `feed.xml`-URL toe in Feedly.

Het standaardmodel is `gemini-3.8-flash`. Een ander model kan zonder codewijziging
worden ingesteld als Actions-variable `GEMINI_MODEL`.

De feed wordt om minuut 7 en 37 van ieder uur opnieuw gebouwd. Feedly bepaalt zelf
wanneer het de feed opnieuw ophaalt; daardoor kan een nieuw artikel daar later
verschijnen dan op de statuspagina. De RSS-feed bevat een TTL-hint van 30 minuten.

## Lokale uitvoering

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python newsfeed.py
python -m unittest discover -s tests -v
```

Zonder `GEMINI_API_KEY` draait de lokale versie bewust met Jaccard-voorselectie en
exact-only-verwijdering. Voor
een volledige lokale run:

```bash
GEMINI_API_KEY="..." GEMINI_MODEL="gemini-3.8-flash" python newsfeed.py
```

## Configuratie

Bronnen en standaardvensters staan in `config/sources.json`. Waar mogelijk is één
brede krantfeed gebruikt om doublures tussen homepage- en rubriekfeeds te vermijden.

Ondersteunde environment variables:

| Variabele | Standaard | Betekenis |
|---|---:|---|
| `GEMINI_API_KEY` | leeg | Activeert semantische beoordeling |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini-model-ID |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Model voor kandidaatclustering |
| `EMBEDDING_DIMENSIONS` | `768` | Aantal dimensies per embedding |
| `EMBEDDING_SIMILARITY_THRESHOLD` | `0.78` | Cosine-grens voor kandidaatclusters |
| `EMBEDDING_BATCH_SIZE` | `20` | Artikelen per embedding-request |
| `EMBEDDING_MAX_NEW_PER_RUN` | `40` | Maximum nieuwe embeddings per run |
| `METADATA_MINIMUM_CHARS` | `80` | RSS-samenvattingen hieronder mogen worden verrijkt |
| `METADATA_MAX_PAGES_PER_RUN` | `30` | Maximum artikelpagina's voor metadata per run |
| `HISTORY_HOURS` | `72` | Maximale ouderdom van artikelen |
| `CLUSTER_WINDOW_HOURS` | `36` | Tijdvenster voor kandidaatvergelijking |
| `PUBLIC_BASE_URL` | GitHub Pages-URL | Basis-URL in de RSS-feed |
| `LOG_LEVEL` | `INFO` | Python-logniveau |

## Uitvoer en privacy

- `public/feed.xml`: de Feedly-feed.
- `public/index.html`: leesbare statuspagina.
- `public/status.json`: machineleesbare bron- en runstatus.
- `.cache/newsfeed`: private, kortlevende Actions-cache voor historie,
  paginametadata en gekwantiseerde embeddings; deze map wordt niet gepubliceerd
  of gecommit.

De software haalt geen volledige (betaalde) artikelen op en stuurt die dus ook
niet naar Gemini. Metadata-ophaling accepteert alleen HTTPS, vooraf toegestane
brondomeinen en publieke IP-adressen, volgt maximaal drie gecontroleerde redirects
en leest maximaal 128 KiB. De workflow heeft alleen leesrecht op de repository;
publicatie verloopt via het afzonderlijke Pages-permission. Externe Actions zijn
op volledige commit-SHA's vastgezet, Python-afhankelijkheden staan met hashes in
`requirements.lock`, en CodeQL en Dependabot controleren nieuwe wijzigingen.

Het repositorysecret wordt alleen als HTTP-header aan Google aangeboden. De
publieke feed en status bevatten geen API-sleutel, persoonlijke leesgeschiedenis
of klikgedrag. Ze bevatten wel bewust openbare nieuwsmetadata en de GitHub-
gebruikersnaam die al in de openbare Pages-URL staat.

Bij een nieuw gratis project kan de embeddingcache zich over meerdere runs vullen
door `EMBEDDING_MAX_NEW_PER_RUN`. Een HTTP 429 betekent dat het gratis quotum of
de tijdelijke snelheidslimiet is bereikt; de run blijft dan veilig werken met de
reeds beschikbare embeddings en lokale tekstvergelijking.
