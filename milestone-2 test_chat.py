
from chat_manager import ChatManager
cm = ChatManager()
print(cm.handle('check_balance', {}))
print(cm.handle('apply_loan', {}))
print(cm.handle('nonexistent', {}))
