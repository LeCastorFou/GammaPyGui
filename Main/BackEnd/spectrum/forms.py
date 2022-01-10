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
    picname = StringField('Name of the picture :  *', validators=[DataRequired()])
    submit = SubmitField('Launch spectrum')
