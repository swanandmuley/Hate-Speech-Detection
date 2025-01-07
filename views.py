import numpy as np
import os

from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm

# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def predictImage(request):
    ####importing libraries
    import os
    import librosa
    import librosa.display
    import time
    import subprocess
    import pickle
    import pandas as pd
    import numpy as np
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    test_image = "." + filePathName
    test_image = test_image[2:]

    print(test_image)

    def mfcc(data, sampling_rate):
        mfcc_feature = librosa.feature.mfcc(y=data, sr=sampling_rate)
        return np.ravel(mfcc_feature.T)

    path = test_image
    data, sampling_rate = librosa.load(path, duration=2.5, offset=0.6)

    # print("Length of data: ", len(data))
    # print("MFrequency Cepstral Coefficients: ", mfcc(data, sampling_rate).shape)

    def get_features(path, duration=2.5, offset=0.6):
        data, sampling_rate = librosa.load(path, duration=duration, offset=offset)
        return mfcc(data, sampling_rate)

    data, sampling_rate = librosa.load(path)
    type(np.ravel(librosa.feature.mfcc(y=data, sr=sampling_rate).T))

    data, sampling_rate = librosa.load(path, duration=2.5, offset=0.6)
    X = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    import pandas as pd
    df = pd.DataFrame(X)
    from gtts import gTTS

    # # load the speech model we saved
    import pandas as pd
    import os
    
    

    loaded_model = pickle.load(open('Implementations/final_hate_speech_UI/rnn.sav', 'rb'))
    with open('rnn.sav', 'rb') as f:
     model = pickle.load(f)
    pickle.dump(model, open('rnn_compatible.sav', 'wb'))
    locs = loaded_model.predict([df[0]])
    print(locs,"this is locs")
    print(os.path.exists('Implementations/final_hate_speech_UI/rnn.sav'))
    aa=""

    m=''
    if locs[0]=="Hate":
        m="Hate"
        aa = ("\N{unamused face}")
    elif locs[0]=="Normal":
        m="Normal"
        aa = ("\N{grinning face}")

    d = ("predicted speech category is {0}".format(m))
    language = 'hi'
    myobj = gTTS(text=d, lang=language, slow=False)
    myobj.save("media/welcome.mp3")
    print('Audio Finish')
    #os.system("media/welcome.mp3")
    abc="media/welcome.mp3"

    context = {
        "given_review": aa,
        "review": m,
        'speech': abc
    }
    return render(request, "result.html", context)



