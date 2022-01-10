import os

class Config():
    # clef de protection obtenue avec :
    # import secret
    # secrets.token_hex(16)
    SECRET_KEY = 'e3d382beb72006c1b138754e940145d7'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'

    TOASTR_TIMEOUT = 4000
    TOASTR_EXTENDED_TIMEOUT = 4000
    TOASTR_POSITION_CLASS = 'toast-bottom-full-width'

    ## config Mail
    MAIL_SERVER = 'smtp.googlemail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    #MAIL_USENAME = os.environ.get('EMAIL_USER')
    #MAIL_USENAME = os.environ.get('EMAIL_USER')

    MAIL_USERNAME = ''
    MAIL_PASSWORD = ''

    UPLOAD_FOLDER = os.getcwd()
