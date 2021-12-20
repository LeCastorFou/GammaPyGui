from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, SelectField, PasswordField, SelectMultipleField, SubmitField, BooleanField, TextAreaField, DateTimeField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, NumberRange
from flask_login import current_user
from wtforms.fields.html5 import DateField
import datetime
#Wtform permet de faire toute les validations
# taille, no empty, email pour que l'input de l'utilisateur soit ok

class UpdateAccountForm(FlaskForm):

    submit = SubmitField("Mettre à jour mes infos")

class RegistrationForm(FlaskForm):

    submit = SubmitField("S'inscrire")

class LoginForm(FlaskForm):

    submit = SubmitField('Se connecter')
