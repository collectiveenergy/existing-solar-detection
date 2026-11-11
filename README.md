# Existing solar check

Flags health center sites whose building already has rooftop solar, using free public aerial photos.

**Inputs:** public HRSA site info only (site number, name, address, location). Nothing else.

**How it works** (`detect_solar.py`)
1. **Building outline:** the [Overture Maps](https://overturemaps.org) building that contains the site point, or the largest building within 40 m. An outline can also be given in the sites file (`footprint_wkt`, lon/lat WKT).
2. **Photo:** the newest [USDA NAIP](https://planetarycomputer.microsoft.com/dataset/naip) aerial photo (about 0.6 m per pixel; retaken every 2-3 years per state).
3. **Panels:** an open-source solar panel segmentation model, DeepLabV3-ResNet101 from Kleebauer et al. (2023), *Multi-Resolution Segmentation of Solar Photovoltaic Systems Using Deep Learning*, Remote Sensing 15, 5687. Weights: [Zenodo 10036926](https://zenodo.org/records/10036926), CC BY 4.0. It runs on the photo as is and enlarged 2x.
4. **Per site:** panel area on the roof, and on open ground within 30 m of it (carports, ground mounts; other buildings' roofs are left out, since panels there most likely belong to a neighbour), share of the roof covered, model confidence, photo date. `has_solar` is true when at least 15 m² of panels and 2% of the roof are found (`MIN_ROOF_M2`, `MIN_ROOF_SHARE`). The picture in `results/chips/` shows the photo, the roof outline (cyan), the 30 m ring (dotted) and the panels found.

**Run:** Actions → *Detect existing solar* → *Run workflow*, pick a sites file in `sites/` and how many parts to split it into (about 100 sites per part; parts run side by side) (columns `site_id, site_name, address, lat, lon`; optional `expected_roof_solar` (yes/no) to score a test). Results are saved to `results/`: `<name>_results.csv`, `<name>_summary.txt` for labeled lists, and pictures in `results/chips/<name>/` (every site for small lists; for big lists only flagged sites, labeled solar sites and test misses).

**Sites files:** `test_sites.csv` (9 sites, Santa Clara/San Mateo; roofs checked against current satellite views). `known_solar_test.csv` (those 9 plus 64 sites and organizations with rooftop solar in a 2025 check, in CA, HI, IL, MA, NJ and WA).

**Flags:** `has_solar` (panels on the roof, outline shrunk 1.5 m); `outline_may_be_carport` (the building outline is more than 90% panels, so the site point is probably on a carport or canopy; check the point); `solar_nearby` (confident, panel-shaped areas on open ground within 30 m; may be another parcel, so check). `naip_years` lists every NAIP year available at the site.

**Limits:** panels installed after the photo date are missed. Dark roofs, skylights and rooftop equipment can cause false alarms. Tall downtown buildings are harder because of lean and shadows. Panels next to the building (`solar_nearby`) may sit on a different parcel; NAIP photos cannot show ownership, so those are flagged for a check, not counted as the site's solar. Check flagged sites against a current satellite view.
