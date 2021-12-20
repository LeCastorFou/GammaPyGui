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


class StartAnalysis(FlaskForm):
    source =  SelectField('Choose a source', choices=[('','')] , validators=[])
    rmin = IntegerField('Run min  *', validators=[],default=1)
    rmax = IntegerField('Run Max *', validators=[],default=200000)

    obs = SelectField('Observatory : ', choices=[('H.E.S.S.','H.E.S.S.'),('CTA','CTA'),('FERMI','FERMI')] , validators=[DataRequired()])
    submit = SubmitField('Launch analysis')
