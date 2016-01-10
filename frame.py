class frame():
    def __init__(self,number,skeleton,frameType,right,left,position,touch):
        self.num=number
        self.data=skeleton
        self.height=0
        self.velocity=0
        self.leftheight=0
        self.leftvelocity=0
        self.frameinter=0
        self.isclose=0
        self.ftype=frameType
        self.leftimg=left
        self.rightimg=right
        self.position=position
        self.touch=touch
        self.standardhand=None
        self.standardlefthand=None
