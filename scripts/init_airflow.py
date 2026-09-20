import os

from airflow.www.app import create_app

app = create_app()
with app.app_context():
    security = app.appbuilder.sm
    username = os.environ['AIRFLOW_ADMIN_USER']
    if security.find_user(username=username) is None:
        user = security.add_user(
            username=username, first_name='Lab', last_name='Admin',
            email='admin@example.local', role=security.find_role('Admin'),
            password=os.environ['AIRFLOW_ADMIN_PASSWORD'],
        )
        if not user:
            raise RuntimeError('Airflow administrator creation failed')
    print(f'Airflow administrator {username} ready')
