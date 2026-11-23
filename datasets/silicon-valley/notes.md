# Silicon Valley dataset notes

`silicon-valley` is a hand-made graph that follows Pied Piper from its founding in March 2014 to PiperNet: the people, companies, products, AIs, places, cities and events around it, with dated employment, moves, funding rounds and events. Every node has a short original description for vector search, and the inside jokes are stored as properties and edges, so a query can find them (`weissman_score`, `born_as`, `net_worth_commas`, `RIVAL_OF` in both directions and more).

An homage to the HBO series Silicon Valley; not affiliated with or endorsed by HBO. All descriptions are original.

## Where the story departs from the series

Where the series and the Grafeo AI memory demo (a replay of Richard founding and moving Pied Piper) differ, the replay wins, so every demo tells the same story.

- **Pied Piper is built on middle-out from the start**: in the series it starts on Richard's earlier compression engine, and middle-out arrives the night before TechCrunch Disrupt (2014-06). The replay calls Pied Piper a compression company built on middle-out, so the dataset does too.
- **Pied Piper moves out of the incubator in September 2014**: it is based at the Hacker Hostel from 2014-03 to 2014-09 and at its own office in Mountain View from 2014-09 on. In the series it works from Erlich's house for several seasons; the replay moves it, and the dataset follows the replay.
- **Dates are months, approximated**: the series rarely states dates, so events are placed in the month their season aired (Richard leaves Hooli and founds Pied Piper in 2014-03, TechCrunch Disrupt in 2014-06). Dates the replay pins are kept exactly.
- **Peter Gregory's sesame seeds sit on his Pied Piper investment**: in the series his reasoning from sesame seeds on burger buns drives a commodity bet. The dataset has no commodity nodes, so the joke is a `note` on his seed round in Pied Piper.
- **Erlich's ten percent is an investment**: his incubator stake, paid for with a room at the Hacker Hostel, is an `INVESTED_IN` edge with round `Incubator stake`.
- **Later seasons are simplified**:
  - Gavin Belson has two `CEO` edges to Hooli: until 2017-06, when he is pushed out, and from 2018-04, when he takes the job back. Jack Barker is Hooli's CEO in between and gives the Hooli-Con keynote in 2017. Gavin's second period has no end date; Hooli's later sale is not modelled.
  - Erlich's departure from the Hacker Hostel in the later seasons is not modelled.
  - Pied Piper's later offices are not modelled, so the Mountain View office is its base from 2014-09 on.
- **Roles are condensed**: each person has the roles the story needs, not every title change in the series.
  - Jack Barker's Hooli roles are `Executive` until 2017-06 and `CEO` from 2017-06 to 2018-04; the start of his time at Hooli is not dated.
  - The Box is first pitched at Pied Piper under Jack Barker and later sold by Hooli as the Signature Edition, so the `the-box` product has `first_pitched_at` set to `Pied Piper`.
- **Some details are kept vague or invented**:
  - Fiona's maker is described only as a robotics startup.
  - Endframe is described only as a rival with its own version of the technology.
  - A few homes and offices are invented for the demo: Bream-Hall on Sand Hill Road, and the cities people live in, which are kept from the Grafeo website's graph.
  - `jian-yangs-fridge.talks` is an invention.

## Modelling notes

- A person can have several `WORKS_AT` edges to one company, one per role and period. For example, Richard is Pied Piper's CEO, then its CTO, then its CEO again.
- A month-only date (`2014-03`) loads as the first of the month. In GQL and Cypher it is a `DATE` with `<field>_precision: 'month'` beside it; in Turtle it is an `xsd:date` with `sv:<field>Precision "month"` on the same subject. Day dates carry no precision, and both sides compare and filter as dates.

## Quoted catchphrases

One short quoted catchphrase per character at most, stored in a `catchphrase` property:

- **Erlich Bachman**: "Aviato".

A few names and phrases from the series are kept as property values rather than catchphrases:

- **Hooli**: `motto` "Making the world a better place".
- **Jack Barker**: `nickname` "Action Jack" and `management_theory` "Conjoined Triangles of Success".
- **Russ Hanneman**: `roi_stands_for` "Radio on internet" and `net_worth_commas` 3.
- **Gavin Belson**: `pledge` "Tethics".
- **Hooli rooftop**: `activity` "rest and vest".

## Regenerating

- After editing `source.json`, run `npm run build` in `datasets/`. It validates the source and rewrites `derived/` (graph JSON, timeline, GQL, Cypher and Turtle). `npm run check` fails when a derived file is out of date.
- After a description changes, also run `npm run embed` to refresh the vectors. It needs the model files; the tools README says where they come from.
