import sqlite3
import matplotlib.pyplot as plt  
def show(labels,data,bin_num=4,level_num=2,level_num_hog=3,para='-s 0 -c 2048 -t 2 -g 0.5'):
    train_labels,train_data,test_labels,test_data=split_data(labels,data,get_classes_with_at_least_num_of_data(labels,num=3))      
    left=[];
    height=[];
    for i in range(192):
        left[i]=i;
        height[i]=data[0][i];
    plt.bar(left, height)  
    plt.ylabel('some numbers')  
    plt.show() 
    
    
def load_data_no_mog(db_file_name="../data/data7.db"):
    RAW_DATA_LENGTH=32
    db = sqlite3.connect(db_file_name)
    labels=[]
    #label2classNo={}
    classNo2Label={}
    classNo=0
    current_data_index=-1
    current_sign_id=0
    current_class_id=0
    data=[]
    for sign_id,frame_id,word_id,lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog in db.execute("Select FrameData.SampleIndex,FrameData.index_ID,signId,SkeletonShoulderLeftX,SkeletonShoulderLeftY,SkeletonShoulderLeftZ,SkeletonShoulderRightX,SkeletonShoulderRightY,SkeletonShoulderRightZ,SkeletonElbowLeftX,SkeletonElbowLeftY,SkeletonElbowLeftZ,SkeletonElbowRightX,SkeletonElbowRightY,SkeletonElbowRightZ,SkeletonHandLeftX,SkeletonHandLeftY,SkeletonHandLeftZ,SkeletonHandRightX,SkeletonHandRightY,SkeletonHandRightZ,SkeletonHeadX,SkeletonHeadY,SkeletonHeadZ,SkeletonHipCenterX,SkeletonHipCenterY,SkeletonHipCenterZ,SkeletonWristLeftX,SkeletonWristLeftY,SkeletonWristLeftZ,SkeletonWristRightX,SkeletonWristRightY,SkeletonWristRightZ,LeftHandHOG,RightHandHog from FrameData, SignSample where FrameData.SampleIndex=SignSample.index_ID ORDER BY  FrameData.index_ID,FrameNumber;"):     
        if current_class_id!=word_id and (not label2classNo.has_key(word_id)):
            classNo+=1
            current_class_id=word_id
            label2classNo[word_id]=classNo
            classNo2Label[classNo]=word_id
        
        if current_sign_id!=sign_id:
            #new sign record
            if current_data_index>=0 and len(data[current_data_index])<54:
                print '!!!!',current_sign_id,len(data[current_data_index])
            current_sign_id=sign_id
            data.append(list())
            labels.append(label2classNo[word_id]) 
            current_data_index+=1
#         print current_sign_id,len(data)
        if lsx:
            data[current_data_index].append([lsx,lsy,lsz,rsx,rsy,rsz,lex,ley,lez,rex,rey,lez,lhx,lhy,lhz,rhx,rhy,rhz,hx,hy,hz,hipcx,hipcy,hipcz,lwx,lwy,lwz,rwx,rwy,rwz,leftHog,rightHog])
#         print data[current_data_index]
    return labels,data,classNo2Label   
    
    
    
if __name__ == '__main__':
    level1=2;
    level2=2;
    rawname='database';
    rawname2='database.db';
    tablename=rawname+'_%r_%r'%(level1-1,level2-1);
#     labels,raw_data=[],[]
    strname='../data/'+tablename;
    databasename='../data/'+rawname2;
    labels,raw_data,classNo2Label=load_data_no_mog(databasename);
    res_file=open('results.txt','w')
    TEMPLATE_THRESHOLD=0.18
    db_file_name="../data/features.db";
    db2 = sqlite3.connect(db_file_name);
    cu=db2.cursor(); 
    cu.execute("select labels,data from "+tablename);
    p=cu.fetchall();
    labels=[];
    data=[];
    for i in range(len(p)):
        ptemp=p[i][1].replace('\\n','\n');
        ptemp=str(ptemp);
        t2 = pickle.loads(ptemp);
        labels.append(p[i][0]);
        data.append(t2);    
    show(labels,data,bin_num=8,level_num=level1,level_num_hog=level2);   
    
    
    
