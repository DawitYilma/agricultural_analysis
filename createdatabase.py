from database import db, Polynomial, RandomForst

db.create_all()

poly = Polynomial(0.012535382, 0.012545528, 0.001878816, 0, 0.002382844, 0, 0.012581169, 0.012673753, 0, 0.007739082, 0.01516428, 0.012612613, 0, 0,0, 0, 0, 0, 0.526296854, 0.309693114, 0.012535382)
db.session.add(poly)
db.session.commit()

rand = RandomForst(-0.764418387,	0.105779259,	0.275373833,	0.153212739,	0.232376326,	0.114285059,	0.282305001,	-0.124457667,	0.020908724,	0.186270461)
db.session.add(rand)
db.session.commit()

print(poly.id)
print(rand.id)

polyall = Polynomial.query.all()
#t = polyall.split(',')
print(polyall)