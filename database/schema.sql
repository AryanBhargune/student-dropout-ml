CREATE TABLE IF NOT EXISTS student_records (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    stream TEXT NOT NULL,
    attendance REAL NOT NULL,
    cgpa REAL NOT NULL,
    fees_paid REAL NOT NULL,
    hostel INTEGER NOT NULL,
    distance_km REAL NOT NULL,
    scholarship INTEGER NOT NULL,
    dropped_out INTEGER NOT NULL,
    academic_year INTEGER NOT NULL
);

