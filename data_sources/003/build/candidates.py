"""Count the candidate pairs reclink2 would score for a master/using pair of tables.

Candidates (task.md, step 2): same `state`, and the same value of at least one of name,
addr, city, zip (empty master values never block). The count is computed with joins and is
not materialized: city-block pairs are counted directly, plus the distinct pairs that agree
on name, addr or zip but not on city. Perfect matches (removed before blocking) are counted
separately and are a small adjustment.

    from candidates import count_candidates
    n_pairs, n_perfect = count_candidates(con, "M", "U")   # table names with name, addr, city, zip, state
"""


def count_candidates(con, M, U):
    city = con.execute(f"""
        SELECT coalesce(sum(m.n * u.n), 0) FROM
          (SELECT state, city, count(*) AS n FROM {M} WHERE city <> '' GROUP BY 1, 2) m
          JOIN (SELECT state, city, count(*) AS n FROM {U} GROUP BY 1, 2) u USING (state, city)""").fetchone()[0]
    other = con.execute(f"""
        WITH mm AS (SELECT row_number() OVER () AS i, * FROM {M}),
             uu AS (SELECT row_number() OVER () AS j, * FROM {U}),
             p AS (
               SELECT mm.i, uu.j FROM mm JOIN uu ON mm.state = uu.state AND mm.name = uu.name
                 WHERE mm.name <> '' AND (mm.city = '' OR mm.city <> uu.city)
               UNION
               SELECT mm.i, uu.j FROM mm JOIN uu ON mm.state = uu.state AND mm.addr = uu.addr
                 WHERE mm.addr <> '' AND (mm.city = '' OR mm.city <> uu.city)
               UNION
               SELECT mm.i, uu.j FROM mm JOIN uu ON mm.state = uu.state AND mm.zip = uu.zip
                 WHERE mm.zip <> '' AND (mm.city = '' OR mm.city <> uu.city))
        SELECT count(*) FROM p""").fetchone()[0]
    perfect = con.execute(f"""
        SELECT count(*) FROM {M} m JOIN {U} u
          ON m.name = u.name AND m.addr = u.addr AND m.city = u.city AND m.zip = u.zip AND m.state = u.state""").fetchone()[0]
    return int(city + other), int(perfect)
