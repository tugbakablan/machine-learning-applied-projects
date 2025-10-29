import sqlite3
import os
def create_database():
    if os.path.exists("students.db"):
        os.remove("students.db")

    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    return conn, cursor

def create_tables(cursor):

    cursor.execute('''
    CREATE TABLE Students (
        id INTEGER PRIMARY KEY,
        name VARCHAR NOT NULL,
        age INTEGER,
        email VARCHAR UNIQUE,
        city VARCHAR)
    ''')

    cursor.execute('''
        CREATE TABLE Courses (
            id INTEGER PRIMARY KEY,
            course_name VARCHAR NOT NULL,
            instructor TEXT,
            credits INTEGER)
        ''')

def main():
    conn, cursor = create_database()
    create_tables(cursor)
    conn.commit()

if __name__ == "__main__":
    main()