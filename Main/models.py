from Main import db, login_manager
from datetime import datetime
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask import current_app
# fonction pour retrouver les utilisateurs par ID  et recharger la Base
# elle peut etre retrouvé sur la doc de flask_login

@login_manager.user_loader
def load_user(user_id):
    try :
        return User.query.get(int(user_id))
    except  Exception:
        return None

class User(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    fbid = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f"User('{self.id}', '{self.email}')"
