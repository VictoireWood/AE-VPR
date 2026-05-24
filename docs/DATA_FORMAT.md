# Data Format

This codebase expects datasets to be passed by command-line path arguments rather than stored in git.

## Training Root

```text
TRAIN_ROOT/
├── Dataframes/
│   └── <dataset_name>.csv
└── Images/
    └── <dataset_name>/
        └── @<year>@<flight_height>@<rotation_angle>@<loc_x>@<loc_y>@.png
```

The CSV columns used by the main dataloaders are:

| Column | Meaning |
| --- | --- |
| `year` | Dataset split or source folder name. |
| `flight_height` | Relative altitude / AGL label in meters. |
| `flight_class` | Altitude bin label. |
| `rotation_angle` | Synthetic or recorded yaw/rotation metadata. |
| `loc_x` | UTM easting or local x coordinate used in filenames. |
| `loc_y` | UTM northing or local y coordinate used in filenames. |

## Test Roots

`test_dc.py` supports several test-set conventions through `--test_set_list`:

| Value | Loader |
| --- | --- |
| `ct01`, `ct02` | `TestDataset` |
| `qd01`, `qd02` | `QingdaoFlightDataset` |

## Altitude Bins

The default fixed-bin setting in the paper is `--M 50`, corresponding to 50 m altitude intervals. Grouped classifiers are controlled by `--N`.

## Files Not To Commit

Do not commit raw imagery, generated tiles, model checkpoints, logs, exported CSV results, or paper PDFs. The `.gitignore` file already excludes the usual locations and extensions.
