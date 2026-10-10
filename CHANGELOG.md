# CHANGELOG

<!-- version list -->

## v0.3.1 (2026-10-10)

### Bug Fixes

- **grading**: Compare a duration as the interval it states
  ([`13f3fe8`](https://github.com/OO-LD/oold-llm-bench/commit/13f3fe81eceb55ce35b5d95db1e601de887aa50b))


## v0.3.0 (2026-10-10)

### Features

- **corpus**: Read an object-valued property as an embedding
  ([`f12fbba`](https://github.com/OO-LD/oold-llm-bench/commit/f12fbbafb1cc7ef0d173f1da184a9a54b0e5d16a))

- **experiments**: Declare a grid over the pages that nest entities
  ([`a14fa16`](https://github.com/OO-LD/oold-llm-bench/commit/a14fa165b627188a4e8dfbae5be0c2e70d7b59c1))

- **graph**: Draw an embedded entity apart from a top-level one
  ([`d004272`](https://github.com/OO-LD/oold-llm-bench/commit/d00427222aa3f20a850c229c318b4d18316c1676))


## v0.2.0 (2026-10-10)

### Bug Fixes

- A property the corpus has never credited is not offered
  ([`3ce0ef2`](https://github.com/OO-LD/oold-llm-bench/commit/3ce0ef27b8a7ec79daff340df3bb94e0118899d4))

### Features

- Every entity records the words it is referred to by
  ([`116c54d`](https://github.com/OO-LD/oold-llm-bench/commit/116c54d18c2f2dceb0af56ff8f061207fd5830b8))

- Name is truth where the document names by title, and steps report both ways of being wrong
  ([`de8e571`](https://github.com/OO-LD/oold-llm-bench/commit/de8e57124da4ba323361404c4c97161d3df3e256))

- Name the training set in every adapter name
  ([`65a948b`](https://github.com/OO-LD/oold-llm-bench/commit/65a948be16fd2081d3144e300da8ee72d3d7003e))

- Offer each slot in the vocabulary the truth came from
  ([`300d0e4`](https://github.com/OO-LD/oold-llm-bench/commit/300d0e4538f76bc49d1dda06bfc969949f9691cd))

- Real text answers what an entity is called, and refuses a lead it did not measure
  ([`1b5e39b`](https://github.com/OO-LD/oold-llm-bench/commit/1b5e39bad6e0c1ab163c9855002b4c4764e36077))

- Reasoning is a run flag, because the providers disagree about it
  ([`49fcde9`](https://github.com/OO-LD/oold-llm-bench/commit/49fcde92c718ca7e1ed16d75aac36bfa1df47211))

- Score one step against the answer the corpus already holds
  ([`0ccb66c`](https://github.com/OO-LD/oold-llm-bench/commit/0ccb66c86edce5165cf214abd416d8479ef52b22))

- The leads load from the Hub, pinned
  ([`e387a30`](https://github.com/OO-LD/oold-llm-bench/commit/e387a3077638449a24880686e2c2d3aa032c236f))

- The property step is shown schema.org's own comments
  ([`d9a88a7`](https://github.com/OO-LD/oold-llm-bench/commit/d9a88a72497c1e48aba7808c4f5c6ff08e5453dd))

- **corpus**: Add sequence, linked-article and Wikidata OO-LD sources
  ([`e054990`](https://github.com/OO-LD/oold-llm-bench/commit/e054990934236ad781f5a281d21ba05a778c6fba))

- **grading**: Read a property against the vocabulary that defines it
  ([`f34856a`](https://github.com/OO-LD/oold-llm-bench/commit/f34856a2345a64348f305e7d7881f26bf8d01f9c))

- **playground**: Carry the plan mention into the graph
  ([`b3a8b51`](https://github.com/OO-LD/oold-llm-bench/commit/b3a8b51e2773631540b7eba5418151aea4056a04))

- **steps**: Declare the step and dedup grids, with their fault buckets
  ([`7fb472f`](https://github.com/OO-LD/oold-llm-bench/commit/7fb472fd6b09b8612b7ccfa133202df87e5dac13))


## v0.1.0 (2026-10-06)

- Initial Release
