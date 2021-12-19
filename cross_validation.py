import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier



def SetData():
    col_names=['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'r16', 'r17', 'r18', 'r19', 'r20', 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27', 'r28', 'r29', 'r30', 'r31', 'r32', 'r33', 'r34', 'r35', 'r36', 'r37', 'r38', 'r39', 'r40', 'r41', 'r42', 'r43', 'r44', 'r45', 'r46', 'r47', 'r48', 'r49', 'r50', 'r51', 'r52', 'r53', 'r54', 'r55', 'r56', 'r57', 'r58', 'r59', 'r60', 'r61', 'r62', 'r63', 'r64', 'r65', 'r66', 'r67', 'r68', 'r69', 'r70', 'r71', 'r72', 'r73', 'r74', 'r75', 'r76', 'r77', 'r78', 'r79', 'r80', 'r81', 'r82', 'r83', 'r84', 'r85', 'r86', 'r87', 'r88', 'r89', 'r90', 'r91', 'r92', 'r93', 'r94', 'r95', 'r96', 'r97', 'r98', 'r99', 'r100', 'r101', 'r102', 'r103', 'r104', 'r105', 'r106', 'r107', 'r108', 'r109', 'r110', 'r111', 'r112', 'r113', 'r114', 'r115', 'r116', 'r117', 'r118', 'r119', 'r120', 'r121', 'r122', 'r123', 'r124', 'r125', 'r126', 'r127', 'r128', 'r129', 'r130', 'r131', 'r132', 'r133', 'r134', 'r135', 'r136', 'r137', 'r138', 'r139', 'r140', 'r141', 'r142', 'r143', 'r144', 'r145', 'r146', 'r147', 'r148', 'r149', 'r150', 'r151', 'r152', 'r153', 'r154', 'r155', 'r156', 'r157', 'r158', 'r159', 'r160', 'r161', 'r162', 'r163', 'r164', 'r165', 'r166', 'r167', 'r168', 'r169', 'r170', 'r171', 'r172', 'r173', 'r174', 'r175', 'r176', 'r177', 'r178', 'r179', 'r180', 'r181', 'r182', 'r183', 'r184', 'r185', 'r186', 'r187', 'r188', 'r189', 'r190', 'r191', 'r192', 'r193', 'r194', 'r195', 'r196', 'r197', 'r198', 'r199', 'r200', 'r201', 'r202', 'r203', 'r204', 'r205', 'r206', 'r207', 'r208', 'r209', 'r210', 'r211', 'r212', 'r213', 'r214', 'r215', 'r216', 'r217', 'r218', 'r219', 'r220', 'r221', 'r222', 'r223', 'r224', 'r225', 'r226', 'r227', 'r228', 'r229', 'r230', 'r231', 'r232', 'r233', 'r234', 'r235', 'r236', 'r237', 'r238', 'r239', 'r240', 'r241', 'r242', 'r243', 'r244', 'r245', 'r246', 'r247', 'r248', 'r249', 'r250', 'r251', 'r252', 'r253', 'r254', 'r255', 'g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15', 'g16', 'g17', 'g18', 'g19', 'g20', 'g21', 'g22', 'g23', 'g24', 'g25', 'g26', 'g27', 'g28', 'g29', 'g30', 'g31', 'g32', 'g33', 'g34', 'g35', 'g36', 'g37', 'g38', 'g39', 'g40', 'g41', 'g42', 'g43', 'g44', 'g45', 'g46', 'g47', 'g48', 'g49', 'g50', 'g51', 'g52', 'g53', 'g54', 'g55', 'g56', 'g57', 'g58', 'g59', 'g60', 'g61', 'g62', 'g63', 'g64', 'g65', 'g66', 'g67', 'g68', 'g69', 'g70', 'g71', 'g72', 'g73', 'g74', 'g75', 'g76', 'g77', 'g78', 'g79', 'g80', 'g81', 'g82', 'g83', 'g84', 'g85', 'g86', 'g87', 'g88', 'g89', 'g90', 'g91', 'g92', 'g93', 'g94', 'g95', 'g96', 'g97', 'g98', 'g99', 'g100', 'g101', 'g102', 'g103', 'g104', 'g105', 'g106', 'g107', 'g108', 'g109', 'g110', 'g111', 'g112', 'g113', 'g114', 'g115', 'g116', 'g117', 'g118', 'g119', 'g120', 'g121', 'g122', 'g123', 'g124', 'g125', 'g126', 'g127', 'g128', 'g129', 'g130', 'g131', 'g132', 'g133', 'g134', 'g135', 'g136', 'g137', 'g138', 'g139', 'g140', 'g141', 'g142', 'g143', 'g144', 'g145', 'g146', 'g147', 'g148', 'g149', 'g150', 'g151', 'g152', 'g153', 'g154', 'g155', 'g156', 'g157', 'g158', 'g159', 'g160', 'g161', 'g162', 'g163', 'g164', 'g165', 'g166', 'g167', 'g168', 'g169', 'g170', 'g171', 'g172', 'g173', 'g174', 'g175', 'g176', 'g177', 'g178', 'g179', 'g180', 'g181', 'g182', 'g183', 'g184', 'g185', 'g186', 'g187', 'g188', 'g189', 'g190', 'g191', 'g192', 'g193', 'g194', 'g195', 'g196', 'g197', 'g198', 'g199', 'g200', 'g201', 'g202', 'g203', 'g204', 'g205', 'g206', 'g207', 'g208', 'g209', 'g210', 'g211', 'g212', 'g213', 'g214', 'g215', 'g216', 'g217', 'g218', 'g219', 'g220', 'g221', 'g222', 'g223', 'g224', 'g225', 'g226', 'g227', 'g228', 'g229', 'g230', 'g231', 'g232', 'g233', 'g234', 'g235', 'g236', 'g237', 'g238', 'g239', 'g240', 'g241', 'g242', 'g243', 'g244', 'g245', 'g246', 'g247', 'g248', 'g249', 'g250', 'g251', 'g252', 'g253', 'g254', 'g255', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b84', 'b85', 'b86', 'b87', 'b88', 'b89', 'b90', 'b91', 'b92', 'b93', 'b94', 'b95', 'b96', 'b97', 'b98', 'b99', 'b100', 'b101', 'b102', 'b103', 'b104', 'b105', 'b106', 'b107', 'b108', 'b109', 'b110', 'b111', 'b112', 'b113', 'b114', 'b115', 'b116', 'b117', 'b118', 'b119', 'b120', 'b121', 'b122', 'b123', 'b124', 'b125', 'b126', 'b127', 'b128', 'b129', 'b130', 'b131', 'b132', 'b133', 'b134', 'b135', 'b136', 'b137', 'b138', 'b139', 'b140', 'b141', 'b142', 'b143', 'b144', 'b145', 'b146', 'b147', 'b148', 'b149', 'b150', 'b151', 'b152', 'b153', 'b154', 'b155', 'b156', 'b157', 'b158', 'b159', 'b160', 'b161', 'b162', 'b163', 'b164', 'b165', 'b166', 'b167', 'b168', 'b169', 'b170', 'b171', 'b172', 'b173', 'b174', 'b175', 'b176', 'b177', 'b178', 'b179', 'b180', 'b181', 'b182', 'b183', 'b184', 'b185', 'b186', 'b187', 'b188', 'b189', 'b190', 'b191', 'b192', 'b193', 'b194', 'b195', 'b196', 'b197', 'b198', 'b199', 'b200', 'b201', 'b202', 'b203', 'b204', 'b205', 'b206', 'b207', 'b208', 'b209', 'b210', 'b211', 'b212', 'b213', 'b214', 'b215', 'b216', 'b217', 'b218', 'b219', 'b220', 'b221', 'b222', 'b223', 'b224', 'b225', 'b226', 'b227', 'b228', 'b229', 'b230', 'b231', 'b232', 'b233', 'b234', 'b235', 'b236', 'b237', 'b238', 'b239', 'b240', 'b241', 'b242', 'b243', 'b244', 'b245', 'b246', 'b247', 'b248', 'b249', 'b250', 'b251', 'b252', 'b253', 'b254', 'b255','NameIMG']
    train_dataset = pd.read_csv("train_data.csv",header=None,names=col_names)

 #   x_train = train_dataset.iloc[:, :-1]
    y_train= train_dataset.iloc[:,769]
    print(train_dataset.shape)
  #  kf=KFold(n_splits=10)
   # rfc = RandomForestClassifier().fit(x_train,y_train)
    #cv=cross_validate(rfc, x_train, y_train, cv=5)
    #print(cv['Test_Score'])

def main():
   SetData()


if __name__=="__main__":
    main()

