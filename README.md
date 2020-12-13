# ccc_deep_blue_project


#### Step 1: Download or Clone the repo
#### Step 2: Follow the below path and open the cmd inside this folder
         1. env > Scripts 
         2. open cmd inside the current folder and write >> activate
         3. After activating the virtual env follow the step 3
#### Step 3: Go inside the proj_deep_blue directory and open cmd(command prompt) and write the below command
         1. python manage.py runserver
                     or
         2. py manage.py runserver
                     or
         3. manage.py runserver
#### Step 4: Now you will receive the one url after writing the above command and just simple copy that url and open it in the internet browser.

## Inportant directory structure for this project

    ccc_deep_blue_project   (Django project)
        |-> env
        |    |-> Scripts
        |          | activate   (batch file)
        |
        |-> proj_deep_blue
                  |-> ccc14   (Django application)
                  |    |-> static
                  |    |     |-> css
                  |    |          | app.css
                  |    |
                  |    |-> templates
                  |    |     | base.html
                  |    |     | home.html
                  |    |     | upload.html
                  |    |
                  |    | admin.py 
                  |    | apps.py
                  |    | imagePrediction.py
                  |    | models.py
                  |    | tests.py
                  |    | views.py
                  |
                  |-> media   
                  |-> proj_deep_blue
                  |        | Settings.py
                  |        | urls.py
                  |        | wsgi.py
                  |
                  |-> yolo_v4   
                  |      | karan_custom.cfg   
                  |      | obj.names  
                  |      | model_best.weights   (Does not exists in the folder you need to add this file while running the project.)
                  |
                  | manage.py   (file for running the project)
