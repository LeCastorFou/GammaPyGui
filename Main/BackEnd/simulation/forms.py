from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed,FileRequired
from wtforms import StringField, SelectField, PasswordField, SelectMultipleField, FloatField, SubmitField, BooleanField, TextAreaField, DateTimeField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, NumberRange
from flask_uploads import UploadSet, IMAGES
from flask_login import current_user
from wtforms.fields.html5 import DateField
from wtforms_sqlalchemy.fields import QuerySelectField
import datetime
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection


class LaunchSimu(FlaskForm):
    livetime = FloatField('Observation time (hour) :  *', validators=[],default=0)
    plong = FloatField('Pointing long :  *', validators=[],default=0)
    plat = FloatField('Pointing lat :  *', validators=[],default=0)
    slong = FloatField('Source longitude :  *', validators=[],default=0)
    slat = FloatField('Source lattitude :  *', validators=[],default=0)
    sIndex = FloatField('Source flux index :  *', validators=[],default=0)
    sFlux = StringField('Flux @ 1 Tev in cm-2 s-1 TeV-1 (ex:1e-11):  *', validators=[],default=0)
    submit = SubmitField('Launch simulation')
