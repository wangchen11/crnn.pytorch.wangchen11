text = "123adbABC你好"
chs = []

for ch in text:
    print(f"{ch} = {hex(ord(ch))} = {ord(ch)}")
    chs.append(ord(ch))

print(f"chs:{chs}")

newText = ""

for num in chs:
    newText += chr(num)
    
print(f"newText:{newText}")