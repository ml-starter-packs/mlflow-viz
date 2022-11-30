run:
	streamlit run app.py

fill:
	python fill.py

install:
	pip install -r requirements.txt

serve:
	mlflow server --port 5005


.PHONY: run fill install serve

