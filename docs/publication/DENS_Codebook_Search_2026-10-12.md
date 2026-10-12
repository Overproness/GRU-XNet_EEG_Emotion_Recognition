# DENS execution/export codebook search and verified contact

12 October 2026, Asia/Karachi. **An author-hosted DENS project and an official dataset inquiry form are now verified. No execution/export codebook was found within the checked public inventories. DENS remains on hold for joins, reservations and supervised experiments.** This search adds documentation evidence; it does not replace the [existing metadata qualification](DENS_Metadata_Qualification_Findings_2026-10-12.md).

After this read-only search, the user separately authorized the inquiry and supplied sender details. [The contact packet](../../results/development/dens_author_contact_2026-10-12/README.md) records one successful form submission at 10:12 Asia/Karachi. References below to no contact describe the search itself; the original audit and inquiry wording remain preserved.

## New author-linked project

The final publication's [Crossref metadata](https://api.crossref.org/works/10.1109/ACCESS.2023.3266804) identifies Sudhakar Mishra's ORCID as `0000-0002-3748-7153`. The [AFDI contributor record](https://api.osf.io/v2/nodes/m8dwz/contributors/?page%5Bsize%5D=100) lists administrator `ezpw7` with the same ORCID marked verified by OSF. That account's [public project inventory](https://api.osf.io/v2/users/ezpw7/nodes/?page%5Bsize%5D=100) links to **[Dataset on Emotion with Naturalistic Stimuli (DENS), OSF n9tbh](https://osf.io/n9tbh/)**. This is an author provenance chain, rather than a name-only attribution to an unrelated repository.

The project was created on 4 November 2023; its node metadata reports modification on 4 June 2026. Its public storage contains three folders: Physiological Data, Papers and Stimuli. There are no public child components or linked nodes. The current checked inventories contain:

| Inventory | Exposed contents | What was read |
| --- | --- | --- |
| Stimuli | 38 movie files, each with filename, version, size and OSF-reported hashes | File metadata only; no movie body |
| Papers | Four author-uploaded article PDFs | Metadata; final IEEE paper body for correspondence verification |
| Physiological Data / Raw Data | Behavioural Data.zip and three large participant ZIP archives | Archive names, sizes and reported hashes only |
| Physiological Data / PreProcessed-Emotion (Group-wise) | EEG_Data.zip | Archive metadata only |
| Physiological Data / PreProcessed-Subject_Wise | Empty | Directory metadata |

The raw participant archive names are `mit003.zip`, `mit004.zip` and `mit014.zip`. The 59,752-byte Behavioural Data.zip was **not downloaded**, nor were any physiological archives. Code or mapping documentation inside an archive cannot be ruled out without a separately bounded inspection; this search makes no claim about unseen archive contents.

The [home wiki](https://api.osf.io/v2/wikis/hwer3/content/) supplies the inquiry link and ODbL/DBCL wording, but also retains an unrelated seven-person motor-task template and an unfinished experiment-description placeholder. Therefore its task, source-format and execution prose cannot authenticate DENS trial semantics.

## Metadata bridge: filename coverage, no title crosswalk

All **20 distinct filename strings** previously qualified from OpenNeuro behaviour first columns appear among the 38 author-project movie entries. This improves the provenance of the numbered-material inventory. It does not establish 38 EEG-recorded materials, independent source titles, practice/main phases, or actual played footage.

An exact comparison of the 38 **OSF-reported full-file SHA-256 values** against the existing 69-entry AFDI metadata inventory found **zero matches**. No ordinal correspondence, transcoding relationship, numbered-key/title mapping or phase assignment was inferred. No movies were downloaded, so the reported movie hashes were not independently checked against body bytes. The separately created [metadata packet](../../results/development/dens_author_project_qualification_2026-10-12/README.md) retains the 38 records, qualified filename coverage and input bindings. The original sealed DENS packet is unchanged.

## Verified inquiry channel and correspondence

The author project's description and home wiki both advertise **[the DENS questions and updates form](https://forms.gle/MzeKuEZEE1ZuPuw98)**. A read-only GET resolves to:

[DENS Dataset — Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSd25CboHi-DBBgEs8sDjkMgWI6sb0Q8mMcMmnb0CoJ-iWSEZA/viewform?usp=send_form).

The visible fields are Name, Mail-Id, Institute, Research Area and Question. **Mail-Id is visibly required.** The form offers Google sign-in to save progress; the read-only page was available anonymously. This does not verify whether submission would require additional steps. No sign-in, field entry, submission or contact-owner action occurred. This form is the most direct verified dataset-specific contact channel; its owner is not independently identified beyond the author-hosted link.

The final paper is also available directly from the author project at **[OSF 27dkm](https://osf.io/27dkm/)**, DOI [10.1109/ACCESS.2023.3266804](https://doi.org/10.1109/ACCESS.2023.3266804). Its 2,405,861-byte PDF exactly matched the OSF inventory SHA-256 `5b4df7c52f7fb0b3c7241f59365e1c5fa3e03a4bb3b62a40517a97fd593ab006`; the DOI and first-page correspondence block were checked. The published corresponding-author options are:

| Author | Published email |
| --- | --- |
| Sudhakar Mishra | rs163@iiita.ac.in |
| Mohammad Asif | pse2017001@iiita.ac.in |
| Uma Shanker Tiwary | ust@iiita.ac.in |

The original [arXiv manuscript](https://arxiv.org/html/2211.02637) independently lists the same addresses. These are authenticated publication addresses; current delivery and account ownership were not tested. They are fallback correspondence routes, separate from the dataset form. No email was sent.

## Other bounded checks

The [pinned OpenNeuro tree](https://api.github.com/repos/OpenNeuroDatasets/ds003751/git/trees/5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c?recursive=1) still contains 318 entries and no execution-code/stimulus directory or behavioural/event JSON codebook. GitHub repository-metadata searches for `DENS Mishra`, `DENS Asif` and `AFDI emotion` returned zero entries; broader `DENS EEG` returned 31, mostly unrelated density/DenseNet projects. The relevant [third-party classifier](https://github.com/Marcushi66/EEG_Emotion_Project) is not original experiment/export evidence. Its fitting results and data were not used.

The [NEMAR DENS mirror](https://nemar.org/dataset/on003751) is a new public distribution lead, but its displayed related identifier points to OpenNeuro 1.0.2. Its [repository](https://github.com/nemarDatasets/on003751/tree/4ecb09ed05a5b469844d7ef8862f32cdaf3ae7b7) has 321 metadata/pointer entries and no exposed original experiment/export codebook. A mirror or converter cannot itself authenticate stimulus identities or original execution. No mirror data bodies were read.

The author-linked AFDI root remains empty, with Films and a manuscript component as its two children. The manuscript component exposes four PDFs, including supplements, but no scripts. Article/publisher supplemental searches produced no authenticated code link. IEEE Xplore presented a JavaScript/robot-verification page, and a bioRxiv supplementary-page attempt was inaccessible. Thus this search **does not establish that no publisher-hosted supplementary artifact exists**. No research-social-network contact or access request was made.

## Reuse statements still conflict

The pinned original README states ODbL for the database and DBCL for individual contents; its dataset-description JSON states CC0. The existing release-bound LICENSE check identifies ODbL text, SHA-256 `52fc126252ca2c5692d39e84050b21fb2342833636c3956f7d417ade30c96e19`. The newly found author wiki repeats ODbL/DBCL, while OSF node metadata has no node license. NEMAR displays CC0 but its README also repeats ODbL/DBCL. These observations do not reconcile the statements or determine legal permissions. The final article's CC BY-NC-ND license concerns the article and must not be substituted for dataset terms.

## Retrieval receipts and next dependency

Selected successful response bindings are recorded below. These are hashes of the retrieved public metadata/documentation bodies, rather than immutable commitments by OSF or GitHub. Bodies were not retained wholesale in this report; future live responses may change. The new movie packet retains only the allowed selected metadata and input bindings.

| Source | Body bytes | Response SHA-256 |
| --- | ---: | --- |
| AFDI contributor API | 7,343 | `1115cf3d71222201454d3e2e51cb9008197e44389ce64af3420865a0c19c75d5` |
| Mishra ezpw7 public-node API | 35,732 | `0cc6012a0757a74f1f053198dbbd452e24703e70e5de6d95dd7dc5ac1db48581` |
| DENS n9tbh node API | 4,892 | `f52c41a7a500ccb390b67e35f006f1ab4886207f65bfde3022e6ecf89f24c0b9` |
| DENS root file inventory | 5,206 | `7c91b7f91336572cdd18bc6bbad111334fb97a62882a0b48506020dba427d553` |
| DENS stimulus inventory | 79,880 | `73634b7bd05863238ea761e147c5a58a6b577dfb13712b77f4a7a2ecf4b1463d` |
| DENS paper inventory | 9,044 | `37e1e6cc9fdd852a76b4dceaed0e2b2fda0911af2438d42633ffa1ebfde4b8b3` |
| DENS raw-archive inventory | 8,658 | `e71f2c87724b9717cc3d3cc44765d32116130e9714b2c664d83e44403175afae` |
| DENS home wiki content | 2,432 | `afea036f603ea5fb5050af6ad5abade924b060a69a8ea2218975a6e67bbb2ca7` |
| DENS inquiry form HTML | 92,582 | `c4b82f15ab93029c8d66283a4cf2520dd4a6a2fb9ae0e6e8ec62f1d223d2ec56` |
| Final-paper Crossref metadata | 6,401 | `43c5c335f823a4d08f0228b268c78d7739bb9ae9902d3e0264bb4edd9ca5055a` |
| Pinned OpenNeuro tree API | 78,734 | `bd8a1359cb6459acb33a0f1810a8ae3d7d56a79a983153150ef19855ffd5b5fe` |

**The concrete next dependency remains the authors' execution/export definition:** numbered filename to source-title/catalogue/version and practice/main map; exact stimulus interval markers and discontinuities; timestamp/duration units; behaviour field scales/missing conventions; physical calibration/reference/preprocessing; and a clarified release reuse statement. The [prepared inquiry](DENS_Metadata_Request_2026-10-12.md) covers these questions and can be routed through the verified form or publication email. It remains unsent.

Only public documentation, directory metadata and the published article were accessed in this continuation. No recording, annotation, behavioural ZIP or movie body was requested; no individual outcome, click value, semantic event field or EEG sample was decoded. No new fit, prediction, participant/material reservation, confirmation access or official research-question change occurred. Authentication of a trial join still requires evidence beyond a matching filename.
