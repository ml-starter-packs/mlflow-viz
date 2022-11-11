
# binder-specific features
with st.sidebar:
    st.markdown('[launch mlflow (may require refresh)](../mlflow/)')
    st.markdown('[edit app (live!)](../lab/)')
    if st.button('Create New Run'):
        os.system('python fill.py')
