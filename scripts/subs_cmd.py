port = input('input port number')
print("copy the following commands and paste to terminal:")

print(f'''
./scripts/copy-key.ps1 wx.blockelite.cn {port} root
ssh root@wx.blockelite.cn -p {port}
cd ~
mkdir ml
cd ml

yes | cp -rf ./scripts/.condarc ~/.condarc
conda install scikit-learn scikit-image tqdm numpy colorama pandas matplotlib albumentations
./scripts/BaiduPCS-Go-Linux/BaiduPCS-Go    login    --bduss mpFQWVCOUJwaGMwQWdMcktqZ05abW1zVE1zaVhYbXJ0dVBxOVc4QlAwbk94SkZoRVFBQUFBJCQAAAAAAAAAAAEAAAB8xRb~emVyb3Roc29uZwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAM43amHON2phM

''') 