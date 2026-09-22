# Persoonlijke AI-nieuwsfeed

Deze repository combineert tien RSS-bronnen tot één feed voor Feedly. De pipeline
verwijdert eerst exacte dubbelen, maakt daarna lokaal goedkope kandidaatclusters
en stuurt uitsluitend die clusters naar Gemini. Artikelen over dezelfde gebeurtenis
blijven staan wanneer ze aantoonbaar extra informatiewaarde hebben.

De gepubliceerde feed komt na het activeren van GitHub Pages beschikbaar op:

`https://miliaan82.github.io/nieuwsfeed/feed.xml`

## Werking

1. GitHub Actions start elke 30 minuten.
2. `newsfeed.py` haalt alleen RSS-feeds op, nooit de achterliggende artikelpagina's.
3. Sportartikelen worden verwijderd op basis van de RSS-categorie, het URL-pad en
   een beperkte titelcontrole. De regels staan in `config/sources.json`.
4. De aggregator voegt de nieuwe items samen met maximaal 72 uur historie uit de
   vorige `public/feed.xml`.
5. Exact gelijke URL's (zonder trackingparameters) en exact gelijke RSS-inhoud
   worden lokaal verwijderd.
6. Een goedkope tekstvergelijking maakt clusters van mogelijke dubbelen.
7. Alleen de titel, RSS-samenvatting en RSS-metadata van die clusters gaan naar
   Gemini.
8. De workflow publiceert `public/feed.xml`, `public/index.html` en
   `public/status.json` via GitHub Pages.

Gemini werkt conservatief: dezelfde gebeurtenis is niet genoeg om iets te
verwijderen. Nieuwe feiten, primaire bronnen, expertise, gevolgen, onzekerheid,
correcties, nuances en wezenlijk andere interpretaties blijven behouden. Andere
toon of framing zonder extra informatie is geen zelfstandig behoudsargument.

### Fail-safe

Als de API-sleutel ontbreekt, Gemini niet bereikbaar is of de respons ongeldig is,
wordt geen enkel mogelijk inhoudelijk duplicaat verwijderd. Alleen de voorafgaande
exacte deduplicatie blijft dan actief. Een bronstoring blokkeert de overige bronnen
niet; nog geldige items uit de vorige feed blijven maximaal 72 uur beschikbaar.

## Eenmalige configuratie op GitHub

1. Open **Settings → Secrets and variables → Actions → New repository secret**.
2. Maak het secret `GEMINI_API_KEY` met de sleutel uit Google AI Studio.
3. Open **Settings → Pages** en kies bij **Source** voor **GitHub Actions**.
4. Open **Actions → Update news feed → Run workflow** voor de eerste handmatige run.
5. Voeg daarna de bovenstaande `feed.xml`-URL toe in Feedly.

Het standaardmodel is `gemini-3.8-flash`. Een ander model kan zonder codewijziging
worden ingesteld als Actions-variable `GEMINI_MODEL`.

## Lokale uitvoering

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python newsfeed.py
python -m unittest discover -s tests -v
```

Zonder `GEMINI_API_KEY` draait de lokale versie bewust in exact-only-modus. Voor
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
| `HISTORY_HOURS` | `72` | Maximale ouderdom van artikelen |
| `CLUSTER_WINDOW_HOURS` | `36` | Tijdvenster voor kandidaatvergelijking |
| `PUBLIC_BASE_URL` | GitHub Pages-URL | Basis-URL in de RSS-feed |
| `LOG_LEVEL` | `INFO` | Python-logniveau |

## Uitvoer en privacy

- `public/feed.xml`: de Feedly-feed.
- `public/index.html`: leesbare statuspagina.
- `public/status.json`: machineleesbare bron- en runstatus.

De software haalt geen volledige (betaalde) artikelen op en stuurt die dus ook
niet naar Gemini. Foutmeldingen en statusbestanden bevatten geen API-sleutel.
