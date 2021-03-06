from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed,FileRequired
from wtforms import TextField, StringField, SelectField, PasswordField, SelectMultipleField, FloatField, SubmitField, BooleanField, TextAreaField, DateTimeField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, NumberRange
from flask_uploads import UploadSet, IMAGES
from flask_login import current_user
from wtforms.fields.html5 import DateField
from wtforms_sqlalchemy.fields import QuerySelectField
import datetime
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection


class StartAnalysis(FlaskForm):
    source =  SelectField('Choose a source', choices=[('','')] , validators=[])
    rmin = IntegerField('Run min  *', validators=[DataRequired()],default=1)
    rmax = IntegerField('Run Max *', validators=[],default=200000)
    obs = SelectField('Observatory : ', choices=[('H.E.S.S.','H.E.S.S.'),('CTA','CTA'),('FERMI','FERMI')] , validators=[DataRequired()])
    submit = SubmitField('Prepare data analysis')

class StartHessAna(FlaskForm):
    source =  SelectField('Choose a source', choices=[('','')] , validators=[])
    analysisName =  StringField('Name your analysis :' , validators=[DataRequired()])
    ra_src = FloatField('Ra source  *', validators=[DataRequired()])
    dec_src = FloatField('Dec source *', validators=[DataRequired()])
    distance = FloatField('Maximun distance of run pointing *', validators=[DataRequired()],default=2)
    rmin = IntegerField('Run min  *', validators=[DataRequired()],default=1)
    rmax = IntegerField('Run Max *', validators=[],default=200000)
    max_evt_offset = FloatField('Maximum event offset *', validators=[DataRequired()],default=2)
    map_size_X = FloatField('Map size X *', validators=[DataRequired()],default=2)
    map_size_Y = FloatField('Map size Y *', validators=[DataRequired()],default=2)
    ana_emin = FloatField('Energy min *', validators=[DataRequired()],default=0.1)
    ana_emax = FloatField('Energy max *', validators=[DataRequired()],default=10.0)
    apply_aeff_mask = BooleanField('Apply AEff mask (for 2D maps) *', validators=[],default=0)
    apply_aeff_mask_value = FloatField('AEff mask percent (for 2D maps) *', validators=[DataRequired()],default=10.0)

    submit = SubmitField('Prepare data analysis')

class StartHess2D(FlaskForm):
    analysis =  SelectField('Select an ongoing Analysis', choices=[('','')] , validators=[])
    ring_inner_radius = FloatField('Ring inner radius *', validators=[DataRequired()],default=0.5)
    ring_width = FloatField('Ring width *', validators=[DataRequired()],default=0.3)
    os_radius = FloatField('Oversampling radius *', validators=[DataRequired()],default=0.05)
    submit = SubmitField('Launch 2D analysis')

class StartHessDataq(FlaskForm):
    analysis =  SelectField('Select an ongoing Analysis', choices=[('','')] , validators=[])
    submit = SubmitField('Launch Data quality')

class SetUpConfig(FlaskForm):
    hessDataPath =  StringField('Path to H.E.S.S fits data folder :' , validators=[])
    ctaIrfsPath =  StringField('Path to CTA IRFS FITS (ex : $PATH/North_z20_50h/irf_file.fits ) :' , validators=[])
    excludedRegionHESS =  StringField('Path to excluded region file (ex : $PATH/myexcludedregions.txt) see file recommention in tutorial :' , validators=[])
    linkWebSummary =  StringField('URL Link to H.E.S.S. Web Summary fo Data Quality :' , validators=[])
    submit = SubmitField('Apply config')
