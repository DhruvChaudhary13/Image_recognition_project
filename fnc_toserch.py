import webbrowser


suggestion=input("enter the search") # here we get the search suggestion 
qerry=suggestion.split() 
len=len(qerry)
# print(len)
# print(qerry)

new_qerry=[]
for k in range(0,len-2):
         new_qerry.append(qerry[k])

# print(new_qerry)

new_str=' '.join(new_qerry)
# print(new_str)

for k in qerry:
    if(k=="youtube"):
      
      webbrowser.open(f"https://www.youtube.com/search?q={new_str}")
    
    elif(k=="google"):
        
        webbrowser.open(f"https://www.google.com/search?q={new_str}")


# webbrowser.open(f"https://www.google.com/search?q={suggestion}") # sugesstion is then passed to the string and the searchred in the browser
