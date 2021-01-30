clear all


%%%%%%%%%%%%%%%%%%%%%%%%参数设置%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=8;
n=204;
k=188;
Nsp=1512;
Nfft=2048;
Ncp=16;
Ns=Nfft+Ncp;
noc=1513;
Nd=1;
M1=2;
M2=16;
sr=250000;
EbNo=0:2:20;
Nfrm=10;
ts=1/sr/Ns;
t=0:ts:(Ns*n*m*2-1)*ts;
fd=100;
h=rayleigh(fd,t);
neb1=zeros(1,Nfrm);
ber1=zeros(1,length(EbNo));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%仿真循环%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:length(EbNo)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%发射机%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    msg=randi([0 2^m-1],Nsp,k);                       %1512*188 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RS编码%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    msg1=gf(msg,m);
    rs=[];
    temp1=rsenc(msg1,n,k).';
    rs=[rs temp1];
    Rs=double(rs.x).';
    for i=1:1512
        Rs_inter(i,:)=JZbian(Rs(i,:));
    end                                             %卷积交织
    msg2=Rs_inter.';
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%卷积编码%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    trel=poly2trellis(7,[171,133]);                         %卷积编码
    tblen=6*7;
    msg3=de2bi(msg2,'left-msb');                    %1512*204=308448个数据 从左到右，从上到下排列
    for i=1:8
        msg4(i,:)=convenc(msg3(:,i).',trel);
    end
    msg5=msg4.';                         
 
    interl=reshape(msg5,2*n*Nsp*m,1);
    p = randperm(2*n*Nsp*m); 
    msg5 = intrlv(interl,p);                                    %比特交织
    msg5=reshape(msg5,2*n*Nsp,m);
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%调制与过信道%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    msg6=reshape(msg5,Nsp,n*m*2);
    data1=pskmod(msg6,M1,pi/4);                          
    data3=zeros(Nfft,n*m*2);
    data3(2:757,:)=data1(757:end,:);
    data3(1293:end,:)=data1(1:756,:);
    
    clear data1;
    
    data3=ifft(data3);
    data4=[data3(Nfft-Ncp+1:end,:);data3];
    
    spow1=norm(data4,'fro').^2/(Nsp*Nd*n*m*2);
    
    data4=reshape(data4,1,Ns*n*m*2);                      %2064*3264
    
    sigma1=sqrt(1/2*spow1/log2(M1)*10.^(-EbNo(ii)/10));
    
   
        dd1=data4;
        hh=h;
        r1=hh.*dd1+sigma1*(randn(1,length(dd1))+1i*randn(1,length(dd1)));
        
        r1=reshape(r1,Ns,n*m*2);
        
        r1=r1(Ncp+1:end,:);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%理想信道估计%%%%%%%%%%%%%%%%%%%%%%%%%
        hh=reshape(hh,Ns,n*m*2);
        hh=hh(Ncp+1:end,:);
        x1=r1./hh;
        x1=fft(x1);
        
        x1=[x1(1293:end,:);x1(2:757,:)];
        x1=pskdemod(x1,M1,pi/4);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%卷积码解码%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        interd=reshape(x1,2*n*Nsp*m,1);
        x1= deintrlv(interd,p);                                %解比特交织
        x1=reshape(x1,Nsp*n*2,m);
        for i=1:8
            x2(i,:)=vitdec(x1(:,i).',trel,tblen,'cont','hard');
        end
        x3=x2.';
        x3=[x3(tblen+1:end,:);randi([0 1],tblen,m)];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%RS解码%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x3=reshape(x3,Nsp*n,m);
        x4=bi2de(x3,'left-msb');
        x5=reshape(x4,n,Nsp).';
        for i=1:Nsp
            x6(i,:)=JZjie(x5(i,:));
        end
        x7=rsdec(gf(x6,m),n,k);
        x7=double(x7.x);
        

        [neb1,temp]=biterr(x7,msg,m);
    
    
    ber1(ii)=sum(neb1)/(Nsp*m*Nd*k);
end

semilogy(EbNo,ber1)
grid on