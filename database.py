import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
######################################
#### SET UP OUR SQLite DATABASE #####
####################################

# This grabs our directory
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.url_map.strict_slashes = False
# Connects our Flask App to our Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'Agricuture_Analytics.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Add on migration capabilities in order to run terminal commands
#Migrate(app,db)

#####################################
####################################
###################################


class Polynomial(db.Model):

    __tablename__ = 'Polynomial'

    id = db.Column(db.Integer, primary_key = True)
    Irrg_No1 = db.Column(db.Float)
    Seed_Not_Improved = db.Column(db.Float)  
    Damage_Yes = db.Column(db.Float)  
    Dreason_Insects = db.Column(db.Float) 
    Dreason_Toolittle_rain = db.Column(db.Float)  
    Dreason_Toomuch_rain = db.Column(db.Float)    
    Dmeasure_Yes = db.Column(db.Float)    
    Dmtype_NonChemical = db.Column(db.Float)  
    Dmchem_Fungicide = db.Column(db.Float)    
    Fert_Yes = db.Column(db.Float)    
    Fert_No = db.Column(db.Float) 
    Ferttype_Natural = db.Column(db.Float)    
    Ferttype_Chemical = db.Column(db.Float)   
    Ferttype_Both = db.Column(db.Float)   
    D22a_DAP = db.Column(db.Float)    
    D22a_Urea_DAP = db.Column(db.Float)   
    D22a_Urea_NPS = db.Column(db.Float)   
    D23_Manure = db.Column(db.Float)  
    Area = db.Column(db.Float)    
    Yield = db.Column(db.Float)   
    Irrg_No2 = db.Column(db.Float)

    def __init__(self,Irrg_No1,Seed_Not_Improved,Damage_Yes,Dreason_Insects,Dreason_Toolittle_rain,Dreason_Toomuch_rain,Dmeasure_Yes,Dmtype_NonChemical,Dmchem_Fungicide,Fert_Yes,Fert_No,Ferttype_Natural,Ferttype_Chemical,Ferttype_Both,D22a_DAP,D22a_Urea_DAP,D22a_Urea_NPS,D23_Manure,Area,Yield,Irrg_No2):
        #self.id = id
        self.Irrg_No1 = Irrg_No1 
        self.Seed_Not_Improved = Seed_Not_Improved   
        self.Damage_Yes = Damage_Yes  
        self.Dreason_Insects = Dreason_Insects 
        self.Dreason_Toolittle_rain = Dreason_Toolittle_rain  
        self.Dreason_Toomuch_rain = Dreason_Toomuch_rain    
        self.Dmeasure_Yes = Dmeasure_Yes    
        self.Dmtype_NonChemical = Dmtype_NonChemical  
        self.Dmchem_Fungicide = Dmchem_Fungicide    
        self.Fert_Yes = Fert_Yes    
        self.Fert_No = Fert_No 
        self.Ferttype_Natural = Ferttype_Natural    
        self.Ferttype_Chemical = Ferttype_Chemical   
        self.Ferttype_Both = Ferttype_Both   
        self.D22a_DAP = D22a_DAP    
        self.D22a_Urea_DAP = D22a_Urea_DAP   
        self.D22a_Urea_NPS = D22a_Urea_NPS   
        self.D23_Manure = D23_Manure  
        self.Area = Area    
        self.Yield = Yield   
        self.Irrg_No2 = Irrg_No2


    def __repr__(self):

        return f"{self.Irrg_No1}  {self.Seed_Not_Improved}  {self.Damage_Yes} {self.Dreason_Insects}  {self.Dreason_Toolittle_rain}  {self.Dreason_Toomuch_rain}  {self.Dmeasure_Yes}  {self.Dmtype_NonChemical}  {self.Dmchem_Fungicide}  {self.Fert_Yes}  {self.Fert_No}  {self.Ferttype_Natural}  {self.Ferttype_Chemical}  {self.Ferttype_Both}  {self.D22a_DAP}  {self.D22a_Urea_DAP}  {self.D22a_Urea_NPS}  {self.D23_Manure}  {self.Area}  {self.Yield}  {self.Irrg_No2}"

class RandomForst(db.Model):

    __tablename__ = 'RandomForst'

    id = db.Column(db.Integer, primary_key = True)
    A = db.Column(db.Float)
    B = db.Column(db.Float)
    C = db.Column(db.Float)
    D = db.Column(db.Float)
    E = db.Column(db.Float)
    F = db.Column(db.Float)
    G = db.Column(db.Float)
    H = db.Column(db.Float)
    I = db.Column(db.Float)
    L = db.Column(db.Float)

    def __init__(self, A, B, C, D, E, F, G, H, I, L):
        #self.id = id
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.H = G
        self.I = I
        self.L = L

    def __repr__(self):

        return f"{self.A}  {self.B}  {self.C}  {self.D}  {self.E}  {self.F}  {self.G}  {self.H}  {self.I}  {self.L}"