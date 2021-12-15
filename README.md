** Memory consumption of Machine Learning models:
Topic - 1.24 GB
Industry - 1 GB
Onhold - 0.1 GB
Reject - .15 GB
Client 214 - 1.25 GB
client 135 1.25 GB

** We will deploy more model as per the bussiness usecase.

** Steps to deploy FastAPi on production.
1. Take pull fastapi Repository.
2. Install the requirements using requirements.txt file which is licated in requirements folder.
3. Change the directory to /etc/systemd/system .
4. Create service named ml_model_api.service to start the fast api.
5. Paste the below Text in ml_model_api.service .

Description=Gunicorn instance daemon to serve API.
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/home/asim/Desktop/contify_fastapi/ml_model_api .
Environment="PATH=/home/asim/.pyenv/versions/fastapi/bin" .
ExecStart=/home/asim/.pyenv/versions/fastapi/bin/gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind unix:ml_model_api.sock -b "0.0.0.0:8082" .

[Install]
WanterBy=multi-user.target

6. Make sure to change the path used in above text.
7. Start the service using sudo systemctl start ml_model_api.
8. Verify if the service running properly by using sudo systemctl status ml_model_api.
9. Install Nginx in system by using sudo dnf install nginx.
10. Change the directory to /etc/nginx/conf.d .
11. Create new file ml_model_api.conf.
12. paste the below text into the file.

server{
       server_name 192.168.0.238;
       location / {
           root /home/asim/Desktop/contify_fastapi/ml_model_api;
           proxy_pass http://127.0.0.1:8082/;
       }
}

13. Change the server name accordingly and save the file.
14. Start the nginx by using sudo systemctl start nginx.
15. check if is running properly by sudo systemctl status nwe will deployginx.
16. To check if nginx is working or not try to acceess url(server_name/docs).












# cfy_ml_api
