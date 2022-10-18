from mydatabase import db
from mydatabase.model import Polynomial_database

db.create_all()

poly = Polynomial_database(0.012535382,0.012545528,0.001878816,0,0.002382844,0,0.012581169,0.012673753,0,0.007739082,0.01516428,0.012612613,0,0,0,0,0,0,0.526296854,0.309693114,0.012535382)

db.session.add(poly)
db.session.commit()

print(poly.id)
