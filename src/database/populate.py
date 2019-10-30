import psycopg2

conn = psycopg2.connect(database="fraud",
                        user="postgres",
                        host="localhost", port="5432")
cur = conn.cursor()

id = 2
fraud = 0.35

insert_vals = [id, fraud]
insert_query = "INSERT INTO fraudstream VALUES \
                (%s, %s)"

cur.execute(insert_query, tuple(insert_vals))
conn.commit()
conn.close()