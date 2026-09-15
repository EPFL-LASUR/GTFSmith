# GTFS

This package has as principal goal to merge GTFS feeds from across Europe to be able to make long-distance public transport routing with the help of [minotor](https://www.minotor.dev/). From this task emerged other functions to work with GTFS that might be useful outside of simply merging GTFS feed.

This README has a goal to present the merge process, as well as the different functions allowing specific transformations of GTFS feeds.

## Global organization of a GTFS feed

GTFS, which stands for _General Transit Feed Specification_ is a (semi-) standardized format allowing public transit agencies to describe their service. Developed originally by Google, it is today widely used in the industry.

A GTFS feed is basically a bunch of .txt files packed in a zip archive. The backbone of a feed is composed of six files:
- stops.txt: list of all the stops deserved by the agency. The stops are usually organized in two level: parent stops describe the whole station, while children stops describe the platforms and are linked to a parent station.
- stop_times.txt: describes for each trip the arrival and departure time at each of its stops, alongside the stop sequence, describing the order in which the stops are deserved. Operates usually at the children stop level, allowing to describe the platform at which the trip is arriving.
- trips.txt: all the trips proposed by the agency. A trip is one leg from a destination to an origin, starting at a given time (e.g. IC1 Geneva-St Gallen, departing at 09h35).
- routes.txt: all the routes proposed by the agency. A route bundles many trips (e.g. route Geneva-St Gallen)
- calendar.txt: describes the dates at which a service is proposed by the agency. A service bundles certain routes and certain trips running on the same days and times.
- calendar_dates.txt: describes usually the exceptions to the services described in calendar.txt (e.g. a certain service is running from 08h00 to 22h00 every day, except on Christmas Day, where it stops at 18h00).

This list is however not exhaustive at all (as said before, GTFS is only a "semi-" standard). Each agency can add many different files, among them the following are often also present in a GTFS feed:
- shapes.txt: describes the exact geographical shape of a route
- frequencies.txt: allows to reduce the size of calendar.txt by describing trips that operate on a regular headway
- transfers.txt: presents the transfer possibilities between two stops (e.g. Geneva (as in the railway station), and Geneva, place Cornavin (the bus platforms in front of the station))

To add a bit to the fun, some agencies do not represent their service the same way in the different files. SNCF for example does not use a calendar.txt file to describe their long-distance services, but only the calendar_dates.txt file. Some agencies do not rely on the parent-children hierarchy and have only parent stations. Other agencies use this hierarchy differently, where children stops do not represent the platforms but kind of servies (e.g. the SNCF differentiates in the children stops between OUIGO, Lyria, TER and Intercités).

Moreover, the different agencies may represent some stops slightly differently. Stops where both rail and bus are proposed might be represented as two stops (e.g.Montparnasse and Monparnasse Gare routière in the SNCF feed) or just as one (Montparnasse in the Swiss feed).

All these little differences between the feeds make it somewhat interesting to combine them. The following shows how this package works around these challenges to obtain one global feed that can be parsed into [minotor](https://www.minotor.dev/), thus permitting public transport routing on a larger scale. For more information about GTFS, the [GTFS Websites'](https://gtfs.org/) [Technical Documentation](https://gtfs.org/documentation/overview/) and the [GitHub Repo](https://github.com/google/transit) are excellent sources.

## Merging GTFS feeds

The code for merging GTFS feeds is to be found in the `gtfs_merge.py` file. The steps to merge two GTFS feeds are the following.

### 1. Merge stops

The one thing two GTFS feeds might have in common are the stops they represent (e.g. Paris Gare de Lyon and Zürich HB are both in the swiss and the french GTFS feeds). The stops are therefore the main key to merge two feeds.

In order to identify the stops that are represented in two feeds, a combination of geographical and semantical proximity is used. The function `_match_parent_stations()` uses `cKDTree` from the package `scipy.spatial` to search, for each stop in the first feed, stops in the second feed that are at a specific distance from the stop in the first feed. This search is done only on Parent Stops, which represent the stations (and not the platforms). This distance is one of the main parameter to adapt when merging GTFS feeds. The default is set to 250 m, however, it has been shown that, when looking at long-distance travel, a threshold of 600 m is needed to ensure that each stops across all the feeds are correctly merged. At the metropolitan scale, a smaller threshold might however be needed.

A distance score is then computed as:
    $$dist_{score} = max(0, 1 - \frac{dist}{dist_{threshold}})$$

Using `ratio` from `rapidfuzz.fuzz`, a name similarity score is computed as well. The final score is given by:
    $$score = 0.25 * nameSimilarity + 0.75 * dist_{score}$$

The stop with the highest score of the second feed is then mapped to the stop of the first feed.

The function `_match_child_stops()` then matches the platforms of two matched parent stops, but only if their platform codes are the same. Otherwise, the children of the parent stop of the second feed are just added to the other children from the first feed. `_merge_stop_hierarchy()` merges the mapped parent stops and the mapped children stops, while `merge_stops()` wraps the whole mapping and merging operation in one function.

### 2. Merge trips

#### Expand frequencies into trips

In this version, it has been chosen to not keep a frequencies file in the final GTFS feed. The trips represented in the `frequencies.txt` file must therefore be expanded to the `trips.txt` and `stop_times.txt` files.

For each trip in `frequencies.txt`, a total amount of $(\frac{endTime - startTime - 1}{headway} + 1)$ are created, and the representative trips in `trips.txt` and `stop_times.txt` are deleted.


#### Match trips

Once the stops have been merged, trips present in both feeds can be matched. In order for two trips to be matched, they need to fulfill three conditions:
- The ordered stop_id sequence must be identical
- The column stop_sequence is identical
- Arrival and departure times at the stops differ by at most the specified time tolerance.

Candidates are detected by a hashed trip signature based on `stop_sequence` and `stop_id`. In order to avoid an extreme unlikely hash collision, the unhasehd ordered sequence of two candidates is checked before trusting the match.


#### Match services

Instead of matching the services afterwards and passing over the whole merged `trips.txt` file to replace the merged services, they are matched directly and only the added trips from the second feed need to be updated. This substantially saves memory usage and computation time.

The matching of service relies on two criterias:
- Their trips have been matched
- The effective operating dates are identical

The second criteria supposes that, for each service, the effective service dates are extracted from the files `calendar.txt` and `calendar_dates.txt`, which is done by the function `_service_dates()`.

#### Merge

Once the trips and services matched, the trips and the stop times can be merged. All the trips and stop times from the first feed are preserved, and the ones that are only present in the second feed are added and their service_ids are updated.

The whole operation is wrapped cleanly in the function `merge_trips()`.

### 3. Merge routes

Once the trips have been merged, the routes can be matched as well. A route pair is considered a match by `_match_routes()` if trips belonging to those routes have been identified as identical.$

The unmatched routes from the second feed are added to the routes of the first feed, while the matched one are deleted from the second feed.

### 4. Merge agencies

Based on the merged routes, agencies present in both feeds are merged. The underlying assumption in this step is that a route present in both feeds will also be operated by the same agency in both feeds. This assumption might not be completely true for all feeds. However, it has been considered as acceptable, as the question of which agency is operating which line is more of a political/economical question than a technical question, and does not need to be fully answered in order to do some routing, which is, at the end of the day, the main objective of this package

### 5. Merge services

The matching of the services has already been [detailled earlier](#match-services). To do this operation however, the `calendar.txt` and `calendar_dates.txt` files needed to be extended to obtain the effective operating dates. The reverse operation needs to be done in order to obtain new `calendar.txt` and `calendar_dates.txt`, which is done by `_date_to_calendar()`.

### 6. Merge transfers

Once all the stops, trips, stop times, routes and services have been merged, the last important step are the transfers. Transfers might be used to specify transfer times between two nearby station (e.g. Genève Cornavin, the railway station, and Genève, gare Cornavin, the bus stations at the front of the station), or between platforms of the same station (e.g. between platform 5 and 35 in Zürich HB).

As the children of the second feed are added to the ones of the first feed, they need to obtain the same transfer times as the ones from the first feed. The transfer time and transfer type are deduced from the transfer times and types of the first feed.

The same goes for inter-station transfers, where the most observed pattern of combined transfer time and transfer type is choosen for the newly added children.

### 7. Merging multiple GTFS feeds

The function `merge_multiple_gtfs()` allows to merge multiple feeds with only one command, instead of repeating the `merge()` function.

An authoritative needs to be chosen. The IDs of this feed will be the overall dominant IDs. Each feed in the provided list is then added one after the other to the dominant feed. A feed listed first will therefore be dominant to a feed listed second, and the last feed of the list will be dominated by all the other feeds.

In order to save RAM, the function deletes the loaded feeds as soon as they have been merged with the authoritative feed.

If an output path is provided, the resulting feed is then written as a .zip file to said path.

## Operations on GTFS feeds

The task of merging different GTFS feeds across Western Europe led to some side-quests, as not all the feeds are specified in the exact same way. Therefore, some functions have been created and used in a pre-processing step.

This functions can however be useful independantly of the task of merging GTFS feeds, and are therefore presented in the following.

### 1. Intermetropolitan transfers

Some important European cities may have different stations (e.g. Paris, Berlin, Milano,...). These stations are connected by local public transportation services, which are important connectors, especially if the stations are terminal stations and are not connected by long-distance services, like it is the case in Paris.

Obtaining the GTFS feeds of all the local public transport providers is however a long and uncertain task. In order to address this issue, transfers are created and added to the `transfers.txt` file, thus modelling the services provided by the local PT agencies.

A buffer of a specified distance is build around each station, and these buffers are then merged. For each station in a same blob, a transfer is generated under two conditions:
- the blob needs to contain a minimum number of stops
- the distance between the two stations cannot exceed a specified value

This rules are set in order to ensure that the transfers are generated in urban areasa, where it is reasonable to expect good PT coverage, allowing for seamingless transfers between these two stations.

The transfer time is then computed using the distance between the two stations, a mean public transport speed, and an added transfer penalty.

The value for the parameters are specified in the table below:

Parameter | Value
--- | :---: 
Buffer radius | 5 km
Min stations/blob | 3
Max. distance betw. stations | 10 km
Mean TP speed | 20 km/h
Transfer penalty | 10 min

All these factors can be adapted, especially when the scale is not long-distance travel across Western Europe

### 2. Merge duplicated physical stops

Some feed have two or more stop IDs for stations which are physically the same. Examples are:
- Paris Montparnasse ("Paris Montparnasse Hall 1 - 2" - "Paris Montparnasse Vaugirard")
- Mulhouse gare ("Mulhouse" - "Mulhouse gare centrale")

The merging of stops [presented earlier](#1-merge-stops) supposes however that physical stops only appear once in every feed. This function therefore checks for each parent station in a feed if there is another parent station in a specified radius.

If it is the case, the first occurence is kept, the children of the second occurence are remapped to the first stop, and the references to the second occurence are remapped as well.

The search radius has been set to 300 m. Again, this is reasonable on the level of long-distance travel, however in a more urban context, this value should be lowered. Passed this distance of 300 m, [intermetropolitan transfers](#1-intermetropolitan-transfers) can be generated to illustrate possible connections.

### 3. Filter GTFS

#### 3.1 Filter per mode

Some feeds (such as the swiss or the netherland's) are aggregated feeds at the national level, and contain therefore long-distance services, but also local and regional bus services. Connections by ship or cable can also be represented.

As delightful it is to obtain this data open-source, it might not be useful for every purpose, and uses a lot of memory. GTFS objects can therefore be filtered according to a specific route type once loaded, reducing their size and drastically improving computation time.

All the references to the routes which are not corresponding to the specified route type are deleted from the GTFS object, reducing its size drastically (especially `stop_times.txt` can become substantially lighter). The memory is also cleaned using the garbage collector `gc.collect()`.

#### 3.2 Filter per geographical area

If only the transport system in a given region/metropolitan area is of interest, it might be useful to clip the GTFS feed to said area, thus reducing the memory usage of the feed. A bounding box in EPSG:4326 is to be given, and all the stops within that bounding box are kept.

The stop times containing stops within that bounding box are then kept, as well as the corresponding trips, routes, services and transfers.

Again, `gc.collect()` is used to clean the memory in the process, allowing to filter big GTFS feed on a computer with less RAM.

### 4. Prefix stop IDs

The stop IDs in different feeds might have the same structure (i.e. just a string of numbers). This might lead to collision when merging the stops. To prevent this, this function adds the specified prefix to each stop_ids and updates all the references (`stops.txt`, `stop_times.txt` and `transfers.txt`).

### 5. Create parent stations

Some providers (Renfe, Trenitalia) do not use the hierarchy of parent and children stops detailled [at the beginning](#global-organization-of-a-gtfs-feed) of this README. Instead, each stop is listed in `stops.txt` as a child, and without parent station (*note*: the stops represent physical stations, not platforms, at least in the Renfe and the Trenitalia feed). [Minotor](https://www.minotor.dev/) however relies on the parent-child hierarchy. Therefore, this hierarchy is artificially produced by creating a parent for each stop and assigning it to said stop.

### 6. Add missing parent stations

This function is quite similar to the one [creating parent stations](#5-create-parent-stations). The difference however is that it is used on feed where the parent-children hierarchy exists, but is not fully implemented, i.e. some platforms have no reference to a parent station.

Again, as [minotor](https://www.minotor.dev/) relies on this hierarchy, a parent station is needed. For stop IDs where the parent station field and the location type is NA, a parent station is therefore created. Its ID is built from the IDs of the children (*note*: this has been tailored on the austrian ÖBB GTFS feed, it may need to be adapted for other feeds requesting such operation), its coordinates is the mean of the platforms coordinates and its name is taken from the children names.