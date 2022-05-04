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

class LaunchSpectrum(FlaskForm):
    source =  SelectField('Choose an ongoing analysis', choices=[('','')] , validators=[])
    #source =  SelectField('Choose a source', choices=[('','')] , validators=[])
    #analysisName =  StringField('Name of spectrum :' , validators=[DataRequired()])
    index = FloatField('Index : ', validators=[],default=2)
    index2 = FloatField('Second index : ', validators=[],default=2)
    amplitude = StringField('Amplitude (cm-2 s-1 TeV-1) example : 1e-12 ', validators=[],default='1e-12')
    reference = FloatField('Amplitude reference : ', validators=[],default=1)
    lambdat = FloatField('lambda (TeV-1) : ', validators=[],default=0.1)
    ebreak = FloatField('E break (TeV) : ', validators=[],default=3)
    alpha = FloatField('alpha : ', validators=[],default=2.3)
    beta = FloatField('beta : ', validators=[],default=0.2)

    submit = SubmitField('Launch spectrum')
