from django.shortcuts import render, redirect
from django.http import HttpResponse
import requests
import json
import subprocess
import os
from query_app.utils import *
from .models import *
from django.http import JsonResponse
from django.views import View
from django.shortcuts import render


def index(request):

    print(request.GET)

    if request.method == 'POST':

        data = request.POST

        print(data)

        codebase = process_repo_util(url = "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0.git", model = CodeList)
        # convert_codebase_to_embeddings(codebase = codebase, model = Embedding, collection_name = 'vina-gpu')
        # pdb.set_trace()
        query = "test"# data['query']
        
        # joined_text = get_best_matches(query = query, model = Embedding)
        # response = query_chatgpt(query = query, codebase = joined_text)

        return JsonResponse({'response': query})
    
    else:
        return render(request, 'index.html')

class YourView(View):
    def get(self, request, *args, **kwargs):
        # pdb.set_trace()
        parameter = kwargs.get('parameter')
        print(parameter)
        joined_text = get_best_matches(query = parameter, model = Embedding)
        response = query_chatgpt(query = parameter, codebase = joined_text)
        # Do something with the parameter value (e.g. query the database for an object with a matching value)
        return JsonResponse({'response': response})



def process_repo(request):

    if request.method == 'POST':

        pdb.set_trace()

        repo_url = data['repo']

        codebase = process_repo(repo_url = repo_url, model = CodeList)

        return HttpResponse(content='Successfully processed the URL', status = 200)
