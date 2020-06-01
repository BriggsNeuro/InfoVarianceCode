%This code receives m-sequence response data from neurons and computes
%information in bits in addition to Maximum entroy and Response entropy
%estimates. It requires the original m-sequence stimulus (series of 1s and
%0s indicating luminance of pixels in noise stimulus grid) that you used to
%obtain the data. It asks you to determine the peak frame and center pixel
%in the receptive field and then it searches through the m-sequence for
%repeats of the "ideal" stimulus in the center of the receptive field in
%order to generate noise entropy estimates from these repeats. It also
%loops through muliple calculations based on different word sizes,
%following the methods of Strong et al, 1998 and similar to Lu et al,
%2001, in order to generate estimates of entropy taking into account word
%size and limitations due to amount of data. Written by Farran Briggs in
%Jan. 2020 with substantial help from Luke Shaw who generated bits/spike
%calculation code.

[path,file]=uigetfile; %read in your data file
load([file path]);
prompt1 = {'Mseq structure #s for No LED', 'Enter Channels','AorBs (1 or 2)?','Bin Size (ms)','ON or OFFs (1 or 0)?'};
ztitle = 'Input';  % Here "structure numbers" specify data from different stimulus runs, in this code the stimulus runs are two conditions: With or Without LED activation of CG feedback
dims = [1 50];
definput1 = {'7 11 11', '5 3 5','1 1 1','8','1 0 0'};
answer1 = inputdlg(prompt1,ztitle,dims,definput1);
numcells = length(str2num(answer1{1})); %You can analyze multiple cells from the same recordings if you have them
rows = str2num(answer1{1}); %these are the structure numbers, you will need to keep track of which condition corresponds to which structure number
chans = str2num(answer1{2}); %these are the recording electrode channels
AorBs = str2num(answer1{3});    % this is in case you have two units on the same channel, As and Bs
binlength=str2num(answer1{4}); %msec - once you settle on a good bin size for you, don't change this
ONorOFFs = str2num(answer1{5}); %this identifes the response of the receptive field center as On or Off - important for selecting the correct pixel sequences for the noise entropy calculation
load Sequence %this is the file containing the actual m-sequence used to generate your visual stimulus. It MUST match the stimulus you used to generate the data.

for aa=1:numcells
    row = rows(aa);
    chan = chans(aa);
    AorB = AorBs(aa);
    ONorOFF = ONorOFFs(aa);
    if length(data(row(1)).gratcyclets)>1 %we save time stamps within each data structure row that correspond to that stimulus run. Here, gratcyclets has multiple entries depending on the number of loops through the m-sequence marking the start of each.
        mseq1dur=data(row(1)).gratcyclets(2)-data(row(1)).gratcyclets(1);
    else
        mseq1dur=327.5962; %this is a cheat that is probably only applicable for our lab, but you may want to set this to the average duration of your m-sequence stimulus run in case a time stamp is missing.
    end
    FPT = 2; % Frames per term that you used to collect your data
    monitorRefreshRateHz = 16383 / mseq1dur * FPT; %16383 is the number of frames in our m-sequence, you will need to adjust this for yours.
    termLengthMs = round((1000 / monitorRefreshRateHz),3);
    displaydur = 1000 * mseq1dur;
    displaynum = round((displaydur) / (FPT * termLengthMs));
    edge = (0:(FPT * termLengthMs):displaydur);
    FrameTriggerTimes = (0:(FPT * termLengthMs):displaydur); %get the time stamps for each m-sequence frame that was displayed - you will need this to calculate noise entropy later.
    if AorB == 1  %Grab the actual neuronal STAs, which have already been calculated and saved to the appropriate row of the data structure. You will need to generate these beforehand.
        kA = data(row(1)).kernelAs(:,:,chan); %separate kernels are calculated for unit As and unit Bs, so make sure you grab the correct kernel.
    else kA = data(row(1)).kernelBs(:,:,chan);
    end
    KernelS.Mseq0 = reshape(kA(:,2),16,16); %make a spatiotemporal STA map from the kernel and display the 4 frames preceding each spike.
    KernelS.Mseq1 = reshape(kA(:,3),16,16);
    KernelS.Mseq2 = reshape(kA(:,4),16,16);
    KernelS.Mseq3 = reshape(kA(:,5),16,16);
    
    scrsz = get(groot,'ScreenSize'); %plot the spatiotemporal STA so that you can select the frame with the peak response
    figure('OuterPosition',[1 scrsz(4)/3 scrsz(3)/1.25 scrsz(4)/3])
    subplot(1,4,4), imagesc(KernelS.Mseq0)
    hold on
    subplot(1,4,3), imagesc(KernelS.Mseq1)
    hold on
    subplot(1,4,2), imagesc(KernelS.Mseq2)
    hold on
    subplot(1,4,1), imagesc(KernelS.Mseq3)
    
    name = 'Enter frame for peak response';
    prompt2 = {'Peak Resp Frame'};
    numlines = 1;
    defaultparams = {'3'}; %choose the peak frame as "1-4" from left to right
    params = inputdlg(prompt2,name,numlines,defaultparams);
    peakFrame = str2double(params{1}); %find the pixel with the maximum On or Off response depending on your choice of peak frame.
    switch peakFrame
        case 1
            if ONorOFF == 1
                [a,b] = max(max(KernelS.Mseq3));
                [c,f] = max(KernelS.Mseq3(:,b));
            end
            if ONorOFF == 0
                [a,b] = min(min(KernelS.Mseq3));
                [c,f] = min(KernelS.Mseq3(:,b));
            end
            time2spk(aa) = 70; %this is just for your records in case you want to keep track of the peak frame time to spike
        case 2
            if ONorOFF == 1
                [a,b] = max(max(KernelS.Mseq2));
                [c,f] = max(KernelS.Mseq2(:,b));
            end
            if ONorOFF == 0
                [a,b] = min(min(KernelS.Mseq2));
                [c,f] = min(KernelS.Mseq2(:,b));
            end
            time2spk(aa) = 50;
        case 3
            if ONorOFF == 1
                [a,b] = max(max(KernelS.Mseq1));
                [c,f] = max(KernelS.Mseq1(:,b));
            end
            if ONorOFF == 0
                [a,b] = min(min(KernelS.Mseq1));
                [c,f] = min(KernelS.Mseq1(:,b));
            end
            time2spk(aa) = 30;
        case 4
            if ONorOFF == 1
                [a,b] = max(max(KernelS.Mseq0));
                [c,f] = max(KernelS.Mseq0(:,b));
            end
            if ONorOFF == 0
                [a,b] = min(min(KernelS.Mseq0));
                [c,f] = min(KernelS.Mseq0(:,b));
            end
            time2spk(aa) = 10;
    end
    Cntrpix(aa,:) = [b,f]; %compute the location of the center receptive field pixel in order to generate the noise entropy sequence below
    clear row chan AorB kA KernelS ONorOFF
end

for zz=1:numcells % loop through this whole program for each neuron to analyze in the opened data structure
    row = rows(zz); %this is the row you input to start (i.e. condition #1)
    row(2) = row(1) + 1; %this is the row for the partner condition (i.e. condition #2) that goes with #1 above
    chan = chans(zz);
    AorB = AorBs(zz);
    ONorOFF = ONorOFFs(zz);
    nbinoptions= [2 4 8 10]; % number of bins (word length); the code will loop through each of these to create the entropy estimates
    bin=binlength/1000; % binlength in seconds
    mseqdurfull = mseq1dur * length(data(row(1)).gratcyclets); % the code will also segment the data a few different ways, so needs the full duration
    X = zeros(1,256);
    X = reshape(X,16,16);
    X(Cntrpix(zz,1),Cntrpix(zz,2)) = 1; % mark the center receptive field pixel from the STA
    X = reshape(X,1,256); % convert this to the correct position in the m-sequence
    Peakpixel = find(X==1);
    pixseq = Sequence(:,Peakpixel); %find all parts of the sequence with the correct luminance in the center receptive field
    if ONorOFF == 0
        prefseq = [1 1 1 0 0 0]; %generate the preferred sequence (transition to preferred pixel luminance)
    elseif ONorOFF == 1
        prefseq = [0 0 0 1 1 1];
    end
    frames = strfind(pixseq',prefseq); %find all instances of this preferred sequence in the m-sequence
    trigtimes = FrameTriggerTimes(frames) + 3*(FPT * termLengthMs); %find the associated time stamps for the start of these sequences - here we include 6 frames
    seqdur = length(prefseq) * 20 / 1000; %in seconds
    
    entout(zz).where=[path '/' file]; %start making an output data structures including the file, channel, bin size, word length, etc
    entout(zz).index=[row(1) chan AorB; row(2) chan AorB];
    entout(zz).wordsize = nbinoptions * binlength;
    wordsizes = entout(zz).wordsize ./1000;
    xaxis = 1./wordsizes; %following Strong et al calculation of estimates based on values / wordsize
    
    for r = 1:length(nbinoptions) %final loop that gives info as a function of length T (wordsize)
        combos=[]; frall=[]; ME=[]; CE=[]; frNE=[]; bn=[]; RE=[]; NE=[];
        nbins = nbinoptions(r);
        combos=dec2bin(0:(2^nbins-1),nbins)-'0';
        
        for z=1:2 % one row of vals for each condition
            repeattrigtimesall=[];
            freq=data(row(z)).samplefreq; %the sampling frequency of the original data is saved into the data structure row
            for m = 1:length(data(row(z)).gratcyclets) %this grabs the time stamps for each of the sequences used for the noise entropy calculation
                repeattrigtimesall = cat(2,repeattrigtimesall,(trigtimes/1000 + data(row(z)).gratcyclets(m)));
            end
            repeattrigtimesall(find(repeattrigtimesall > (data(row(z)).gratcyclets(1) + mseqdurfull - 0.2))) = []; %exclude data after the stimulus ends
            
            % do whole Info (bits) calculation and extrapolation with fractionated data
            for d=1:4 % number of times we will fractionate the data
                frate=[]; nskp=[]; bntemp=[]; retemp=[]; netemp=[]; frNEtemp=[]; Itemp=[];
                for p = 1:d
                    rast=[]; rastvals=[]; bincount=[]; binrast=[]; words=[]; cntr=[]; s=[]; counter=[]; 
                    rast2=[]; binrast2=[]; repeattrigtimes=[]; nekeeptemp=[]; strt=[]; mseqdur=[]; fr=[];
                    mseqdur = mseqdurfull / d;
                    if length(data(row(z)).gratcyclets) < 4
                        strt = data(row(z)).gratcyclets(1) + (mseqdur * (p-1)); %analyze the appropriate segment of the data
                    else
                    strt = data(row(z)).gratcyclets(p);
                    end
                    repeattrigtimes = find(repeattrigtimesall > strt & repeattrigtimesall < strt + mseqdur); %analyze the approprate repeat sequences within the segment under analysis
                    rast=zeros(round(freq*mseqdur),1); %preload rast length of segment in a zeros array
                    if AorB == 1 %AorB As; this gets the actual spikes from the unit As
                        rastvals=find(data(row(z)).unitAts(:,chan)>strt & data(row(z)).unitAts(:,chan)<(strt+mseqdur)); %find spike time indices greater than start and less then strt+dur
                    elseif AorB ==2 %AorB Bs; this gets actual spikes from the unit Bs
                        rastvals=find(data(row(z)).unitBts(:,chan)>strt & data(row(z)).unitBts(:,chan)<(strt+mseqdur)); %find spike time indices greater than start and less then strt+dur
                    end
                    rast(round(freq*data(row(z)).unitAts(rastvals,chan)),1)=1; %make rastval indices into 1's
                    frate(p)=size(rastvals,1)/mseqdur; %general firing frequency
                    
                    for i=1:floor(size(rast,1)/(bin*freq)) %for the amount of bins in the data - this walks along the raster in bin sized chunks
                        if ismember(1,rast(((i-1)*round(bin*freq)+1):i*round(bin*freq))) %if there's a 1 in there make it a 1
                            binrast(i)=1;
                            bincount(i)= nnz(rast(((i-1)*round(bin*freq)+1):i*round(bin*freq))); %rather than counting any number of spikes as 1, this gives you spike count for bin containing spike(s)
                        else
                            binrast(i)=0;
                            bincount(i)=0;
                        end
                    end
                    bntemp(p) = length(find(bincount(:)>1))/length(bincount); %this tells us the percentage of bins that have more than 1 spike, useful for making sure you have the correct bin size
                    
                    for i=1:(size(binrast,2)-(nbins+1)) %for the total number of words in binrast with 1 bin step size
                        words(i,:)=binrast(1,(i-1)+1:(i-1)+nbins); %splits binned raster into words
                    end
                    for j = 1:size(combos,1)  % Do probabilities for response entropy
                        repeated=0;
                        for i = 1 : size(words,1)
                            repeated=repeated + prod(double(combos(j,:)==words(i,:))); % so this compares word pattern to actual bin string, if all bins match, you get all 1's, multiply together (prod), you get 1. if one mismatch (0) then you get 0. double converts logical to numbers.
                        end
                        cntr(j,1)=repeated;
                    end
                    counter=sortrows(horzcat(cntr,combos),1,'descend');
                    counter(:,1)=counter(:,1)/sum(counter(:,1)); % this is a really useful thing to look at
                    
                    % Calculate Response Entropies
                    s=counter(:,1).*log2(counter(:,1));
                    s(find(isnan(s)))=0;
                    retemp(p) = -sum(s); % response entropy from Strong et al
                    
                    %noise entropy
                    rast2 = zeros(freq*(seqdur),length(repeattrigtimes));
                    for k = 1:length(repeattrigtimes)
                        spikestemp=[]; spikes=[];
                        if AorB == 1 %this grabs the spikes just within the repeating segments of the preferred pixel sequence (for unit As)
                            spikestemp = find(data(row(z)).unitAts(:,chan) > repeattrigtimes(k) & data(row(z)).unitAts(:,chan) < repeattrigtimes(k) + seqdur);
                            spikes = ceil(freq*(data(row(z)).unitAts(spikestemp,chan) - repeattrigtimes(k)));
                        elseif AorB == 2 %same as above for unit Bs
                            spikestemp = find(data(row(z)).unitBts(:,chan) > repeattrigtimes(k) & data(row(z)).unitBts(:,chan) < repeattrigtimes(k) + seqdur);
                            spikes = ceil(freq*(data(row(z)).unitBts(spikestemp,chan) - repeattrigtimes(k)));
                        end
                        rast2(spikes,k) = 1; %make rastval indices into 1's
                        fr(k) = length(spikes)/seqdur; %get the firing rate for each sequence
                    end
                    rast2 = reshape(rast2,(k*(freq*seqdur)),1); %reorganize the rasters for each segment for the entropy calculation
                    for i=1:floor(size(rast2,1)/(bin*freq)) %for the amount of bins in the data - this walks along the raster in bin sized chunks
                        if ismember(1,rast2(((i-1)*round(bin*freq)+1):i*round(bin*freq))) %if there's a 1 in there make it a 1
                            binrast2(i)=1;
                        else
                            binrast2(i)=0;
                        end
                    end
                    binrast2 = reshape(binrast2,k,(seqdur/bin));
                    for n = 1:k
                        counter2=[]; s2=[]; cntr2=[]; words2=[];
                        for i=1:(size(binrast2,2)-(nbins+1)) %for the total number of words in binrast with 1 bin step size
                            words2(i,:)=binrast2(n,(i-1)+1:(i-1)+nbins); %splits binned raster into words
                        end
                        for j = 1:size(combos,1)  % Do probabilities for noise entropy
                            repeated=0;
                            for i = 1 : size(words2,1)
                                repeated=repeated + prod(double(combos(j,:)==words2(i,:))); % this compares word pattern to actual bin string, if all bins match, you get all 1's, multiply together (prod), you get 1. if one mismatch (0) then you get 0. double converts logical to numbers.
                            end
                            cntr2(j,1)=repeated;
                        end
                        counter2 = sortrows(horzcat(cntr2,combos),1,'descend');
                        counter2(:,1)=counter2(:,1)/sum(counter2(:,1)); % this is a really useful thing to look at
                        s2=counter2(:,1).*log2(counter2(:,1)); %entropy calc.
                        s2(find(isnan(s2)))=0;
                        nekeeptemp(n) = (-sum(s2));
                    end
                    netemp(p) = mean(nekeeptemp); %average together noise entropies per sequence within the segment
                    frNEtemp(p) = mean(fr); %average the firing rates for the same sequences
                    Itemp(p) = retemp(p) - netemp(p); %for the segment, information is Response Entropy - Noise Entropy
                end %loop for each fraction of the data

                %Do probabilities and Max entropy using mean frate for
                %loops per fraction from Pryluk et al, 2019
                probcombos=combos;
                probcombos(find(probcombos==0))=1-(mean(frate)*bin); %prob of filling bin with 0 given the mean firing rate
                probcombos(find(probcombos==1))=mean(frate)*bin; %prob of filling bin with 1 given the mean firing rate
                probcombos=prod(probcombos,2); % word (bin combo) joint probabilities
                entout(zz).word(r).mecounter{z,d}=sortrows([probcombos combos],1,'descend'); %enter Max Entropy values into output structure
                
                nskp=probcombos.*log2(probcombos);
                nskp(find(isnan(nskp)))=0;
                ME(z,d)= -sum(nskp); %Compute Maximum entropy as above and save for appropriate wordsize and segment
                frall(z,d) = mean(frate); %same for firing rate, firing rate from noise entropy, etc
                frNE(z,d) = mean(frNEtemp);
                bn(z,d) = mean(bntemp);
                RE(z,d) = mean(retemp);
                NE(z,d) = mean(netemp);
                CE(z,d) = RE(z,d)/ME(z,d);
                Info(z,d) = mean(Itemp); 
            end %loop for all fractions
        end
        entout(zz).ME(:,:,r) = ME;
        entout(zz).RE(:,:,r) = RE;
        entout(zz).NE(:,:,r) = NE;
        entout(zz).frall(:,:,r) = frall;
        entout(zz).frNE(:,:,r) = frNE;
        entout(zz).bn(:,:,r) = bn;
        entout(zz).CE(:,:,r) = CE;
        entout(zz).Info(:,:,r) = Info;
        
        figure %this computes the estimated entorpy values for Information and ME based on wordsize - note these use linear fits to estimate Y-intercept
        for z = 1:2
            subplot(1,2,z), scatter([1:4], Info(z,:)/wordsizes(r),'k')
            hold on
            subplot(1,2,z), scatter([1:4], ME(z,:)/wordsizes(r),'m')
            hold on
            fitI = fit([1:4]',(Info(z,:)/wordsizes(r))','poly1');
            fitME = fit([1:4]',(ME(z,:)/wordsizes(r))','poly1');
            subplot(1,2,z), plot(fitI,'k')
            hold on
            subplot(1,2,z), plot(fitME,'m')
            hold on
            entout(zz).trueInfo(r,z) = fitI.p2;
            entout(zz).trueME(r,z) = fitME.p2;
        end
    end
    
    figure %this computes the extrapolated Information and ME based on segmentation - notes these use linear fits to estimate Y-intercept
    for z = 1:2
        MEout(z,:) = entout(zz).trueME(:,z);
        Iout(z,:) = entout(zz).trueInfo(:,z);
        subplot(1,2,z), scatter(xaxis, MEout(z,:),'m')
        hold on
        subplot(1,2,z), scatter(xaxis, Iout(z,:),'k')
        hold on
        fitME2 = fit(xaxis(1:3)',MEout(z,1:3)','poly1');
        subplot(1,2,z), plot(fitME2,'m')
        hold on
        fitI2 = fit(xaxis(1:3)',Iout(z,1:3)','poly1');
        subplot(1,2,z), plot(fitI2,'k')
        hold on
        entout(zz).extrapI(z) = fitI2.p2;
        entout(zz).extrapME(z) = fitME2.p2;
    end
    clear chan AorB ONorOFF
end
entoutNew = entout;
clearvars -except entoutNew
[path,file]=uigetfile;
load([file path]);
save(path);