
# binder-specific features
with st.sidebar:
    st.markdown('[launch mlflow](../mlflow/)')
    st.markdown('[edit app (live!)](../lab/tree/app.py)')
    if st.button('Create New Run'):
        os.system('python fill.py')
