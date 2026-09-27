# Phase 2 Cleaning & Normalization Audit Report

**Dataset Root**: `/Users/swapnil/Documents/ML/student_resource/dataset`

## 1. File Statistics & Verification

| File | Original Rows | Cleaned Rows | Row Count Pass | ID Preservation Pass | Orig Missing Addrs | Clean Missing Addrs | Countries Found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1` | `2,206,821` | `2,206,821` | `True` | `True` | `0` | `0` | `India, US` |
| `train_source2` | `5,034,616` | `5,034,616` | `True` | `True` | `168,967` | `168,967` | `India, US` |
| `train_source3` | `5,285,603` | `5,285,603` | `True` | `True` | `175,916` | `175,916` | `India, US` |
| `test_source1` | `1,732,544` | `1,732,544` | `True` | `True` | `0` | `0` | `France, India, US` |
| `test_source2` | `4,887,273` | `4,887,273` | `True` | `True` | `129,408` | `129,408` | `France, India, US` |
| `test_source3` | `5,082,316` | `5,082,316` | `True` | `True` | `136,098` | `136,098` | `France, India, US` |

## 2. Normalization Rules Applied

- **Unicode Normalization**: Applied `NFKC` across all string fields.

- **Case & Whitespace**: Converted to lowercase, collapsed multiple whitespace into single space, stripped leading/trailing spaces.

- **Punctuation**: Replaced all Unicode punctuation and symbols with space, preserving all letters, numbers, and non-English scripts.

- **Whole-Token Suffix Normalization**: Normalized business suffixes as whole tokens (e.g., `corp` -> `corporation`, `pvt` -> `private`, `ltd` -> `limited`, `inc` -> `incorporated`, `co` -> `company`). Substrings inside words were untouched.

- **Whole-Token Address Normalization**: Normalized address abbreviations as whole tokens (e.g., `rd` -> `road`, `st` -> `street`, `ave` -> `avenue`, `blvd` -> `boulevard`, `apt` -> `apartment`, `no` -> `number`).

- **Address Numbers Extraction**: Extracted space-separated numeric sequences into `address_numbers`.

- **Country Normalization**: Standardized `US`, `India`, `France`, and any other country string into title/upper cased formats.

- **Missing Address Handling**: Kept original missing address rows intact, setting `business_address_clean`, `address_numbers`, and `address_tokens` to empty string `""`.


## 3. Sample Normalization Examples

| File | Entity ID | Original Name | Cleaned Name | Significant Tokens | Original Address | Cleaned Address | Address Numbers | Country | Country Clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train_source1` | `S1-925783039` | `Orelee's Barbershop` | `orelee s barbershop` | `orelee s barbershop` | `1795 Westchester Drive, High Point, NC` | `1795 westchester drive high point nc` | `1795` | `US` | `US` |
| `train_source1` | `S1-773889195` | `Prime Money` | `prime money` | `prime money` | `17560 Ellis Road, Tahlequah, OK` | `17560 ellis road tahlequah ok` | `17560` | `US` | `US` |
| `train_source1` | `S1-377745466` | `B+ Retail Inc` | `b retail incorporated` | `b retail` | `1712 Montebello Avenue, Phoenix, AZ` | `1712 montebello avenue phoenix az` | `1712` | `US` | `US` |
| `train_source1` | `S1-133037285` | `Christ Chapel` | `christ chapel` | `christ chapel` | `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD` | `2100 cameron drive unit apartment g dundalk md` | `2100` | `US` | `US` |
| `train_source1` | `S1-755362802` | `Prabhav Business Center` | `prabhav business center` | `prabhav business center` | `797, Lake Town Block A, Kolkata, Howrah, West Bengal` | `797 lake town block a kolkata howrah west bengal` | `797` | `India` | `India` |
| `train_source1` | `S1-851869949` | `Custom Wealth Services LLC` | `custom wealth services llc` | `custom wealth services` | `OH, Columbus, 5559 Orville Avenue` | `oh columbus 5559 orville avenue` | `5559` | `US` | `US` |
| `train_source1` | `S1-785847572` | `Consulting Nyasa Nursing Private Limited` | `consulting nyasa nursing private limited` | `consulting nyasa nursing` | `2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra` | `2505 tower 1 oakwood runwal greens mulund goreagon link road near fortis hospital bhandup west mumbai maharashtra` | `2505 1` | `India` | `India` |
| `train_source1` | `S1-27541239` | `Nexus Anchor Rain` | `nexus anchor rain` | `nexus anchor rain` | `1111 Church Street, Unit 2007, Nashville, TN` | `1111 church street unit 2007 nashville tn` | `1111 2007` | `US` | `US` |
| `train_source1` | `S1-629417405` | `Moore Bitwise Inc` | `moore bitwise incorporated` | `moore bitwise` | `337 Oakland Avenue, Michigan City, IN` | `337 oakland avenue michigan city in` | `337` | `US` | `US` |
| `train_source1` | `S1-22305073` | `Dermatology Green Medicine` | `dermatology green medicine` | `dermatology green medicine` | `294 Meadowcreek Drive, Unit Unit 2, Village Of Pewaukee, WI` | `294 meadowcreek drive unit unit 2 village of pewaukee wi` | `294 2` | `US` | `US` |
| `train_source2` | `S2-166376419` | `राम मार्केटिंग प्राइवेट लिमिटेड` | `राम मार्केटिंग प्राइवेट लिमिटेड` | `राम मार्केटिंग प्राइवेट लिमिटेड` | `KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi` | `kh number 570 13 new delhi west delhi delhi` | `570 13` | `India` | `India` |
| `train_source2` | `S2-764573417` | `-- Holloway Peak Inc Seafood` | `holloway peak incorporated seafood` | `holloway peak seafood` | `105 ELM ST, MORGANTON, NC` | `105 elm street morganton nc` | `105` | `US` | `US` |
| `train_source2` | `S2-639257739` | `आदित्य प्रॉपर्टीज एलएलपी` | `आदित्य प्रॉपर्टीज एलएलपी` | `आदित्य प्रॉपर्टीज एलएलपी` | `G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh` | `g 3 571 gulmohar colony bhopal madhya pradesh` | `3 571` | `India` | `India` |
| `train_source2` | `S2-163963287` | `Summit Inc` | `summit incorporated` | `summit` | `GREENSBORO, NC, 19 1/2 STARDUST TRAIL` | `greensboro nc 19 1 2 stardust trail` | `19 1 2` | `US` | `US` |
| `train_source2` | `S2-49942811` | `Delta Tetlecommunication Inc` | `delta tetlecommunication incorporated` | `delta tetlecommunication` | `914 PIERPONT AVE, CLEVELAND, OH` | `914 pierpont avenue cleveland oh` | `914` | `US` | `US` |