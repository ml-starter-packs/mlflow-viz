
# start mlflow in background
import subprocess
subprocess.run(["make", "serve"])

# binder-specific features
with st.sidebar:
    st.markdown('[launch mlflow](../mlflow/)')
    st.markdown('[edit app (live!)](../lab/)')
    if st.button('Create New Run'):
        os.system('python fill.py')
