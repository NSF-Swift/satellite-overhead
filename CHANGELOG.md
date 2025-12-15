# CHANGELOG



## v0.8.2 (2024-04-18)


### Fix

* fix: Convert bandwidth data read from FrequencyData.csv to MHz. ([`28091ea`](https://github.com/NSF-Swift/satellite-overhead/commit/28091ea603533b343ab26fde98024cdaca160ae5))



## v0.8.1 (2024-04-04)



## v0.8.0 (2024-04-03)


### Build

* build: Add python 3.11 and 3.12 to the testing matrix. ([`09a1d55`](https://github.com/NSF-Swift/satellite-overhead/commit/09a1d55ec2bd26998c4c4943426390d8874eac66))


### Feature

* feat: Add regex name filtering for satellites list. Available via filter_name_regex() ([`818d007`](https://github.com/NSF-Swift/satellite-overhead/commit/818d007e2d529d1048686f5a55f391c641f75f1f))


### Fix

* fix: Add validation for Sopp class parameters. ([`4bdf787`](https://github.com/NSF-Swift/satellite-overhead/commit/4bdf787817d466a75ee56f2753343577c0c99d04))


### Refactor

* refactor: Remove dependency on pytz and instead use the timezone support now included as part of the standard library since python 3.9. ([`4b8ad2a`](https://github.com/NSF-Swift/satellite-overhead/commit/4b8ad2ac71596076a82505a14d94a5bc34d965e5))


### Test

* test: Add validation tests for Sopp parameter validation. ([`ba077db`](https://github.com/NSF-Swift/satellite-overhead/commit/ba077db3a13b7c1ab4d6266db67e4b7ac390b422))



## v0.7.1 (2024-03-15)


### Fix

* fix: Use default_factory for initialization of default field in dataclass. ([`8a38464`](https://github.com/NSF-Swift/satellite-overhead/commit/8a38464ab725d2754b307686dc0bc8fe55e01085))



## v0.7.0 (2024-03-15)



## v0.6.0 (2024-03-14)


### Build

* build: Remove exclusion of main.py from package construction as it was removed. ([`b9d2fd7`](https://github.com/NSF-Swift/satellite-overhead/commit/b9d2fd7e66391cf300728d7a4c9fa1e48c4db921))


### Feature

* feat: Add TardyS4 resrevation request generator. ([`51a540f`](https://github.com/NSF-Swift/satellite-overhead/commit/51a540f7c3b42e0ddbaa48feaafe3c0ffa7b3ea8))

* feat: Refactor filters to use fns instead of lambdas for more flexibility. Add ability to add filters with add_filter versus making filters. Add additional optional context parameter to filter functions for more complex filtering cases. ([`78f96c0`](https://github.com/NSF-Swift/satellite-overhead/commit/78f96c0b53f00350f0cbb0dfcc93d89c634e420e))

* feat: Add min_altitude field in RuntimeSettings which allows setting a minimum altitude for satellites to be considered visible. Useful for when 0 degrees does not represent the visible horizon. ([`51d12f5`](https://github.com/NSF-Swift/satellite-overhead/commit/51d12f5edc84d4d6cf738258beeb76e1138a353b))


### Refactor

* refactor: Use Sopp instead of Main in sopp.py ([`0646ccf`](https://github.com/NSF-Swift/satellite-overhead/commit/0646ccfb9b7f38b25f3c8003f282517bf9e2a07f))

* refactor: Remove Main and replace with Sopp. ([`78eb60b`](https://github.com/NSF-Swift/satellite-overhead/commit/78eb60be2151f1ec2b397d31e369a190390f9fc1))

* refactor: Remove Astropy code/dependencies as it is no longer used. ([`cb6b5ff`](https://github.com/NSF-Swift/satellite-overhead/commit/cb6b5fff497d8c9a6781a4a6fa059e0e796f45de))


### Test

* test: Add test for new add_filter() method in ConfigurationBuilder. ([`2d1daa3`](https://github.com/NSF-Swift/satellite-overhead/commit/2d1daa303956b57cdd51a39ccef43e80ddcf21a6))

* test: Add tests for minimum_alt runtime setting. ([`5ebbd57`](https://github.com/NSF-Swift/satellite-overhead/commit/5ebbd57e1363471cf434ccefe055df640cdf5c7c))



## v0.5.1 (2024-02-28)


### Fix

* fix: Provide a default antenna_trajectory for use in get_satellites_above_horizon if user supplies none. ([`c2f0e45`](https://github.com/NSF-Swift/satellite-overhead/commit/c2f0e451429fef80f87bdc04315963b31fef6898))


### Performance

* perf: Improve performance of get_satellites_above_horizon with refactor. ([`4a8a0d7`](https://github.com/NSF-Swift/satellite-overhead/commit/4a8a0d789f33aad63c93e4186167327ee0f87c09))



## v0.5.0 (2024-02-14)


### Feature

* feat: Add __str__ impl for Configuration, Reservation, FrequencyRange, Facility and TimeWindow classes. ([`e9806dc`](https://github.com/NSF-Swift/satellite-overhead/commit/e9806dc0a7699d904893db1a1fecf78432cdc73e))

* feat: set_time_window() now accepts datetimes as well as datetime strings. ([`5bf9196`](https://github.com/NSF-Swift/satellite-overhead/commit/5bf9196695d69db64c0b1716f01c9bd90e493e16))


### Fix

* fix: Resolve the requirement for an observation target to be set when building a configuration as an observation target is not necessary if checking for satellites above horizon. ([`b610c19`](https://github.com/NSF-Swift/satellite-overhead/commit/b610c197f0172f5c985c0180de303bd1bc796dea))


### Refactor

* refactor: When None is passed to any filtering function they now return a lambda that always evaluates to True instead of failing. ([`fda8d2f`](https://github.com/NSF-Swift/satellite-overhead/commit/fda8d2fd2103c4977db395e5180a1fa3917a3399))

* refactor: Extract bandwidth and frequency from set_facility and move to seperate method set_frequency_range. ([`8c51e69`](https://github.com/NSF-Swift/satellite-overhead/commit/8c51e699fb15aef547e69143b45fdaaf36cf2fd2))

* refactor: Use a single orbit_is function instead of a seperate function for leo/meo/geo. ([`9c57292`](https://github.com/NSF-Swift/satellite-overhead/commit/9c57292c37934af5b34f4a9218798440767bd0af))



## v0.4.0 (2024-01-25)



## v0.3.1 (2024-01-24)


### Build

* build: Bump Skyfield dependency from 1.45 to 1.47. ([`55e4df0`](https://github.com/NSF-Swift/satellite-overhead/commit/55e4df0776f9660342ad281d8d321ad7d8108dc6))


### Feature

* feat: Include distance, represented in km, for satellite positions. ([`5d5a5b4`](https://github.com/NSF-Swift/satellite-overhead/commit/5d5a5b4e58b29a6918bbc92414172cca769f812b))


### Fix

* fix: Calculate orbits per day instead of orbital period and update filters to reflect for leo/meo/geo. ([`5ba4d45`](https://github.com/NSF-Swift/satellite-overhead/commit/5ba4d45a074d0484e70e35b14b858a9ddfeadaf5))


### Test

* test: Add test for orbits_per_day property on Satellite class. ([`ee60d1f`](https://github.com/NSF-Swift/satellite-overhead/commit/ee60d1ff4e1c533438e0c2d5567f3412608b9d56))



## v0.3.0 (2024-01-18)



## v0.2.0 (2024-01-18)


### Build

* build: correctly name Publish action. ([`6dbfe16`](https://github.com/NSF-Swift/satellite-overhead/commit/6dbfe1683e43e4f858d6f5dbdb936b12111f665e))

* build: Add ci/cd pass/fail badge to README. ([`adf0bd1`](https://github.com/NSF-Swift/satellite-overhead/commit/adf0bd1e4ea5c862531fc665a24c9f55babfd50f))

* build: create CHANGELOG.md and GH Release note templates. ([`d4bfd1d`](https://github.com/NSF-Swift/satellite-overhead/commit/d4bfd1d7c2e51897f7900997e44e769498fee783))

* build: add version file. ([`f49a90e`](https://github.com/NSF-Swift/satellite-overhead/commit/f49a90ef7d3b30a70a9d25b2eece20e8eba260c3))

* build: use dynamic versioning. ([`325ee39`](https://github.com/NSF-Swift/satellite-overhead/commit/325ee393380ee98f6a96ef95ccea5e3c33337317))

* build: use psr to automatically version releases. ([`8e57945`](https://github.com/NSF-Swift/satellite-overhead/commit/8e5794544c7dad5aadfba67d909867872087d8da))


### Documentation

* docs: Update documentation to include information on satellite filtering. ([`65c8ddc`](https://github.com/NSF-Swift/satellite-overhead/commit/65c8ddcfe66634b3774dc8ce23aaa8bebe001434))


### Feature

* feat: Add set_satellites_filter function which takes a Filterer class with filters to later filter the list of satellites based on the applied filters. ([`bbbcf67`](https://github.com/NSF-Swift/satellite-overhead/commit/bbbcf672affb24fb777b51e9df5c6f24dc037bc6))

* feat: Add orbital_period property to Satellite class which provides the orbital period in minutes. ([`f8fb8f2`](https://github.com/NSF-Swift/satellite-overhead/commit/f8fb8f2c294c928af72269cc369b7ad59697c059))

* feat: add ConfigurationBuilder to easily create Configuration objects ([`8301e30`](https://github.com/NSF-Swift/satellite-overhead/commit/8301e30475d5084a69f27ef81e49bece8b03bb42))


### Fix

* fix: resolve error where datetime strings without microsencs caused an error ([`fe65b13`](https://github.com/NSF-Swift/satellite-overhead/commit/fe65b135c7e5d55d392670e4b2ff7baec5e07e8f))


### Refactor

* refactor: Rename dir to satellites_filter. ([`a4851b5`](https://github.com/NSF-Swift/satellite-overhead/commit/a4851b52aaa67995cb630e26a82b60b120265bf2))

* refactor: Rename geo/leo/meo filters to be clearer. ([`d5c380f`](https://github.com/NSF-Swift/satellite-overhead/commit/d5c380f4098de1c1ee923f9570eaabfbffaad4fc))

* refactor: Change filter name to name_does_not_contain_filter for clarity. ([`df2dac6`](https://github.com/NSF-Swift/satellite-overhead/commit/df2dac69fa7dda9031dda39bdb1cc29bc49552c4))

* refactor: Use a generic filtering strategy. ([`2bdb59f`](https://github.com/NSF-Swift/satellite-overhead/commit/2bdb59fb871163bb9b55e4adb8cf01038a1cb86b))

* refactor: Update Main() to use refactored frequency filter. ([`01e92d9`](https://github.com/NSF-Swift/satellite-overhead/commit/01e92d9d708a55fabe0f267c27227954b5cc38a0))


### Style

* style: fix whitespace. ([`0624af9`](https://github.com/NSF-Swift/satellite-overhead/commit/0624af98efc88bdc10658bfa2efd9e0ace4395ab))


### Test

* test: Add tests for satellites filtering ([`51432b0`](https://github.com/NSF-Swift/satellite-overhead/commit/51432b061b20ae95c76b695ddf879a637625daa9))

* test: Add tests for refactored filtering strategy. ([`eec6b83`](https://github.com/NSF-Swift/satellite-overhead/commit/eec6b83306f030ec926778d14f50456e0817d5de))

* test: Update tests to use refactored frequency filter. ([`3b47f7c`](https://github.com/NSF-Swift/satellite-overhead/commit/3b47f7c8d70aa2a86976bf60964d83085205590a))

* test: Update tests to account for filtering refactor. ([`edeede3`](https://github.com/NSF-Swift/satellite-overhead/commit/edeede30ef6f160dfc75c044df98b35889dd312d))



## v0.1.0 (2024-01-09)

