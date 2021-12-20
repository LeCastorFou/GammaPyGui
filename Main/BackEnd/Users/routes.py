from flask import Blueprint
from flask import render_template, url_for, flash, redirect, request, abort
from Main import db, bcrypt, mail
from Main.BackEnd.Users.forms import RegistrationForm, LoginForm, UpdateAccountForm
from Main.models import User
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import os
import pandas as pd
import numpy as np
#import secrets
import os
from PIL import Image
from wtforms.fields.html5 import DateField
import datetime
from datetime import timedelta
from flask import jsonify
import json

usersbp = Blueprint('usersbp',__name__)


@usersbp.route("/register", methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('mainbp.home'))
    form = RegistrationForm()
    # sucess is a bootsrap class
    if form.validate_on_submit():
        try:
            user = create_user(form.email.data.strip() , form.password.data)
            registrerfb = True
        except Exception as inst:
            print('### ERROR ####')
            if inst.args[0] == 'The user with the provided email already exists (EMAIL_EXISTS).':
                Message_fr = "Email existant"
                flash(Message_fr,'error')
            elif inst.args[0] == "Invalid password string. Password must be a string at least 6 characters long.":
                Message_fr = "Le mot de passe doit faire au mois 6"
                flash(Message_fr,'error')
            else:
                Message_fr = "Erreur inconnue"
                flash(Message_fr,'error')

            registrerfb = False
        if registrerfb:
            user_db = User(email=form.email.data.strip(), fbid=user.uid)
            db.session.add(user_db)
            db.session.commit()
            #send_welcome_email(form.email.data,form.email.data.strip(),form.password.data)
            user = {'fbid':[user.uid],'pic':['avatardef.jpg'], 'username':[form.pseudo.data],'adresse':[''],'spot':[''],'sport':[''],'distance_max':[30],'ventmini':[17]}
            user = pd.DataFrame.from_dict(user)
            my_dict = user.to_dict('record')
            flash("Compte actif !", 'success')
            return redirect(url_for('usersbp.Login'))
    return render_template('Users/Forms/Register.html', title='Register', form = form)

@usersbp.route("/login", methods=['GET','POST'])
def Login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = LoginForm()
    if form.validate_on_submit():
        try:
            res = sign_in_WEP(form.email.data, form.password.data)
            if res['registered']:
                user = User.query.filter_by(email = res['email']).first()
                login_user(user,remember=form.remember.data)
                flash("Logged IN!", 'success')
                return redirect(url_for('main.home'))
            else:
                Message_fr = "Mauvais mot de passe ou email"
                flash(Message_fr,'error')
                return redirect(url_for('usersbp.Login'))
        except Exception as inst:
            Message_fr = "Mauvais mot de passe ou email"
            flash(Message_fr,'error')

    return render_template('Users/Forms/Login.html', title='Login', form =form)

@usersbp.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@usersbp.route("/updateAccount", methods=['GET','POST'])
def updateAccount():
    fbid = current_user.fbid
    form = UpdateAccountForm()


    # sucess is a bootsrap class
    if form.validate_on_submit():
        try:
            print('ok')
            print(form.username.data)
            print(request.form.get('autocomplete'))

            if request.form.get('autocomplete') == None or request.form.get('autocomplete') == "":
                adresse = user['adresse']
            else:
                adresse = request.form.get('autocomplete')

            if form.picture.data:
                picture_file = save_picture(form.picture.data)
            else:
                picture_file = user['pic']
            user = {'fbid':[fbid],'username':[form.username.data],'adresse':[adresse],'spot':[form.spot.data],
            'sport':[form.sport.data],'distance_max':[form.distance_max.data],'pic':picture_file,'ventmini':[form.ventmini.data]}
            ## efface les infos précédentes
            user = pd.DataFrame.from_dict(user)
            my_dict = user.to_dict('record')
            flash('Updated!','success')
            return redirect(url_for('usersbp.updateAccount'))
        except Exception:
            flash('Erreur inconnue','error')
            return redirect(url_for('usersbp.updateAccount'))
    elif request.method ==  'GET':
        form.username.data = user['username']
        form.distance_max.data = user['distance_max']
        form.ventmini.data = user['ventmini']
    #mettre les valeurs par default des SelectField
    form.spot.process_data(user['spot'])
    form.sport.process_data(user['sport'])


    return render_template('Users/Forms/UpdateAccount.html', title='Register', form =form, img = user['pic'], adresse = user['adresse'])
