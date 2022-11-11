# Jupyterlab icons can be created here to launch apps
# that are included in the user's image.
c.ServerProxy.servers = {
    "app": {
        "command": ["streamlit", "run", "app.py"],
        "port": 8501,
        "absolute_url": False,
        "new_browser_tab": True,
        "launcher_entry": {
            "title": "App",
        },
    },
    "mlflow": {
        "command": ["mlflow", "server", "--port", "5005"],
        "port": 5005,
        "absolute_url": False,
        "new_browser_tab": True,
        "launcher_entry": {
            "title": "MLflow",
        },
    },
}
