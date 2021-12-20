from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from flask_toastr import Toastr
import os
import socket
from Main.config import Config
from flask_simple_geoip import SimpleGeoIP
from flask_debugtoolbar import DebugToolbarExtension
from flask_restful import Api

# Database
db = SQLAlchemy()
# Bcrypt passwords
bcrypt = Bcrypt()
# adding toastr for banners
toastr = Toastr()
# mail extention
mail = Mail()
api = Api()

# Gerer la connection et les Login
login_manager = LoginManager()
login_manager.login_view = 'usersbp.Login'
# donne la class bootstrap info au message d'erreur necessite l'auth
login_manager.login_message_category = 'info'
login_manager.login_message = u"Vous devez être connecté pour accéder à cette page"
# localisation
simple_geoip = SimpleGeoIP()

#template_dir = os.path.abspath('Main/static/FrontEnd/templates')
template_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)) , 'static/FrontEnd/templates')

def create_app(config_class = Config):
    app = Flask(__name__, template_folder=template_dir)
    app.config.from_object(Config)

    if socket.gethostname() in ['WindProject','CFT-AZURE-LX005'] :
        print('server')
    else:
        app.debug = False
        toolbar = DebugToolbarExtension()
        # authoriser les redirections
        app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
        toolbar.init_app(app)

    db.init_app(app)
    bcrypt.init_app(app)
    toastr.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    simple_geoip.init_app(app)
    api.init_app(app)

    from Main.BackEnd.main.routes import main
    app.register_blueprint(main)
    api.init_app(app)

    from  Main.BackEnd.Users.routes import usersbp
    app.register_blueprint(usersbp)

    return app
