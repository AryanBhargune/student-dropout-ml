import random
import argparse
import sqlite3
import os

DB_PATH = os.path.join("database", "student_dropout.db")

STREAMS = ["CSE", "ECE", "Mechanical", "Civil", "BBA", "BCom"]
GENDERS = ["Male", "Female"]

def calculate_dropout(attendance, cgpa, fees_paid, hostel, distance):
    risk_score = 0

    if attendance < 60:
        risk_score += 0.4
    if cgpa < 6.0:
        risk_score += 0.3
    if fees_paid < 100:
        risk_score += 0.2
    if distance > 30 and hostel == 0:
        risk_score += 0.1

    return 1 if random.random() < risk_score else 0


def generate_student(year):
    age = random.randint(17, 24)
    gender = random.choice(GENDERS)
    stream = random.choice(STREAMS)
    attendance = round(random.uniform(40, 95), 2)
    cgpa = round(random.uniform(4.0, 9.8), 2)
    fees_paid = random.choice([60, 75, 100])
    hostel = random.choice([0, 1])
    distance = round(random.uniform(1, 50), 2)
    scholarship = random.choice([0, 1])

    dropped_out = calculate_dropout(
        attendance, cgpa, fees_paid, hostel, distance
    )

    return (
        age, gender, stream, attendance,
        cgpa, fees_paid, hostel,
        distance, scholarship,
        dropped_out, year
    )


def insert_data(records):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
    INSERT INTO student_records (
        age, gender, stream, attendance,
        cgpa, fees_paid, hostel,
        distance_km, scholarship,
        dropped_out, academic_year
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    cursor.executemany(query, records)
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--records", type=int, default=500)

    args = parser.parse_args()

    data = [generate_student(args.year) for _ in range(args.records)]
    insert_data(data)

    print(f"{args.records} records inserted for year {args.year}.")


if __name__ == "__main__":
    main()
