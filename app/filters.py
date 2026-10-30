# filters.py
# RecordingFilter: which recordings match the chosen filters and the search text.

def match_equal(value, chosen):
    return value == chosen

def match_part(value, chosen):
    """Mixed recordings have two types: 'S3 + Normal' matches 'S3' and 'Normal'."""
    return chosen in value.split(" + ")

# Open/Closed: each filter is one line (label in the GUI, column in the table, how to match).
# To add a filter, add a line here. Nothing else in the app has to change.
FILTERS = [
    ("Category", "category", match_equal),
    ("Sound type", "sound_type", match_part),
    ("Location", "location", match_equal),
    ("Gender", "gender", match_equal),
]

class RecordingFilter:
    """`chosen` is a dict {filter label: value}, where 'All' means no filter."""

    def __init__(self, table):
        self.table = table

    @staticmethod
    def parts(sound_type):
        """'S3 + Normal' -> ['S3', 'Normal']"""
        return sound_type.split(" + ")

    def options(self, label, column, category):
        """Values for one dropdown. Category always shows everything,
        the others only show what exists in the chosen category."""
        df = self.table
        if label != "Category" and category != "All":
            df = df[df["category"] == category]
        return ["All"] + sorted({part for v in df[column] for part in self.parts(v)})

    def apply(self, chosen, search=""):
        df = self.table
        for label, column, match in FILTERS:
            if chosen[label] != "All":
                df = df[df[column].apply(lambda v: match(v, chosen[label]))]
        text = search.lower().strip()
        if text:
            df = df[df["file"].str.lower().str.contains(text)
                    | df["sound_type"].str.lower().str.contains(text)]
        return df.reset_index(drop=True)

    @staticmethod
    def describe(chosen):
        """Short text for chart titles, e.g. 'Category: Heart, Gender: F'."""
        parts = [f"{label}: {value}" for label, value in chosen.items() if value != "All"]
        return ", ".join(parts) if parts else "all recordings"
