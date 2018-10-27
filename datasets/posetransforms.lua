
local image = require 'image'

local M = {}

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------
function M.getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function M.transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = M.getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------
function M.crop(img, center, scale, rot, res)
    local ul = M.transform({1,1}, center, scale, 0, res, true)
    local br = M.transform({res+1,res+1}, center, scale, 0, res, true)


    local pad = math.floor(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local newDim,newImg,ht,wd

    if img:size():size() > 2 then
        newDim = torch.IntTensor({img:size(1), br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
        ht = img:size(2)
        wd = img:size(3)
    else
        newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2])
        ht = img:size(1)
        wd = img:size(2)
    end

    local newX = torch.Tensor({math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2]})
    local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1) - 1})
    local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1) - 1})

    if newDim:size(1) > 2 then
        newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
    else
        newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
    end

    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        if newDim:size(1) > 2 then
            newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
        else
            newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        end
    end

    newImg = image.scale(newImg,res,res)
    return newImg
end

function M.crop2(img, center, scale, rot, res)
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
           -- Zoomed out so much that the image is now a single pixel or less
           if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
           return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = M.transform({1,1}, c, s, 0, res, true)
    local br = M.transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
    end

    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------

function M.drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local tmpSize = math.ceil(3*sigma)
    local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
    local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 2*tmpSize + 1
    local g = image.gaussian(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    return img
end
--根据坐标点画圆
function M.drawMask(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local  function judge_out_border(x,y)--x坐标,y坐标,ruguo
        if x>0 and y>0 and x<=img:size(2) and y<=img:size(1) then
            return nil--没超出边界
        end
        return 1--超出边界为真
    end

    --待检测样本点x，待检测样本点y，圆心坐标,求出欧式距离
    local function compute_dist(originx,originy,destx,desty)
        return math.sqrt((originx-destx)*(originx-destx)+(originy-desty)*(originy-desty))
    end
    local function judge_dist(originx,originy,destx,desty,R)
        if compute_dist(originx,originy,destx,desty)<R then
            return 1
        else
            return nil
        end
    end
    local R=2
    if judge_out_border(pt[1],pt[2]) then--标记点超出边界啦
        return img
    else
        xl=pt[1]-R--x坐标最小值
        xh=pt[1]+R--x坐标最大值
        yl=pt[2]-R--y坐标最小值
        yh=pt[2]+R--y坐标最大值
        local i,j--图像的大小为64哎
        --print("当前图像大小为："..img:size(1))
        for  i=yl,yh do
            for j=xl,xh do--对应pt1
                if not judge_out_border(j,i) then --没有超出边界
                    if judge_dist(j,i,pt[1],pt[2],R) then
                        img[i][j]=1
                    end
                end
            end
        end
    end
    return img
end
function M.drawHead(img, pt1,pt2)--脖子和头部
    local  function judge_out_border(x,y)--x坐标,y坐标,ruguo
        if x>0 and y>0 and x<=img:size(2) and y<=img:size(1) then
            return nil--没超出边界
        end
        return 1--超出边界为真
    end
    --待检测样本点x，待检测样本点y，圆心坐标,求出欧式距离
    local function compute_dist(originx,originy,destx,desty)
        return math.sqrt((originx-destx)*(originx-destx)+(originy-desty)*(originy-desty))
    end
    local function judge_dist(originx,originy,destx,desty,R)
        if compute_dist(originx,originy,destx,desty)<R then
            return 1
        else
            return nil
        end
    end
    if judge_out_border(pt1[1],pt1[2]) then--标记点超出边界啦
      return img
    end
    if judge_out_border(pt2[1],pt2[2]) then--标记点超出边界啦
        return img
    end
    local pt=pt1
    local R=math.floor(compute_dist(pt1[1],pt1[2],pt2[1],pt2[2])/2)+1
    pt[1]=math.floor((pt1[1]+pt2[1])/2)
    pt[2]=math.floor((pt1[2]+pt2[2])/2)
    xl=pt[1]-R--x坐标最小值
    xh=pt[1]+R--x坐标最大值
    yl=pt[2]-R--y坐标最小值
    yh=pt[2]+R--y坐标最大值
    local i,j--图像的大小为64哎
    for  i=yl,yh do
      for j=xl,xh do--对应pt1
         if not judge_out_border(j,i) then --没有超出边界
            if judge_dist(j,i,pt[1],pt[2],R) then
               img[i][j]=1
            end
         end
      end
    end

    return img
end
-------------------------------------------------------------------------------
-- 将圆之间的点连接起来
-------------------------------------------------------------------------------
function M.drawresult(imgs,nParts,pts)
    local  function judge_border(x,y,border)--x坐标,y坐标,ruguo
        if x>0 and y>0 and x<=border and y<=border then
            return 1--没超出边界
        end
        return nil--超出边界为真
    end
    --将两个点连接起来
    local function drawLine(mask,x1,y1,x2,y2,len)--len是指扩张宽度
        --print("当前坐标点为:x1："..x1..'-Y1:'..y1..'-X2:'..x2..'-Y2:'..y2)
        xdist=math.abs(x1-x2)
        ydist=math.abs(y1-y2)
        if xdist>ydist then
            mask=drawx(mask,x1,y1,x2,y2,len)--x方向跨度比较大，那么按照x方向递增设置坐标点
        else
            mask=drawy(mask,x1,y1,x2,y2,len)--按照y方向递增设置坐标点
        end
        return mask
    end
    function drawx(mask,x1,y1,x2,y2,len)--x方向跨度比较大，按照x方向递增设置坐标点
        local i
        if x2>x1 then--x1递增
            slope=(y2-y1)/(x2-x1)--斜率
            for i=x1 , x2 do
                y=math.floor(slope*(i-x1)+y1)
                for index=y-len,y+len do
                    if judge_border(index,index,mask:size(1)) then
                        mask[index][i]=1
                    end
                end
                --print("中间坐标点为:x:"..i..'-Y:'..y)
            end
        else
            slope=(y1-y2)/(x1-x2)--斜率
            for i=x2 , x1 do
                y=math.floor(slope*(i-x2)+y2)
                for index=y-len,y+len do
                    if judge_border(index,index,mask:size(1)) then
                        mask[index][i]=1
                    end
                end
                --print("中间坐标点为:x:"..i..'-Y:'..y)
            end
        end
        return mask
    end
    function drawy(mask,x1,y1,x2,y2,len)--按照y方向递增设置坐标点,len是指扩张宽度
        local i
        if y2>y1 then
            slope=(x2-x1)/(y2-y1)--斜率
            for i=y1,y2 do
                x=math.floor(slope*(i-y1)+x1)
                for index=x-len,x+len do
                    if judge_border(index,index,mask:size(1)) then
                        mask[i][index]=1
                    end
                    --print("中间坐标点为:x:"..x..'-Y:'..i)
                end
            end
        else
            slope=(x1-x2)/(y1-y2)--斜率
            for i=y2 , y1 do
                x=math.floor(slope*(i-y2)+x2)
                for index=x-len,x+len do
                    if judge_border(index,index,mask:size(1)) then
                        mask[i][index]=1
                    end
                    --print("中间坐标点为:x:"..x..'-Y:'..i)
                end
            end
        end
        return mask
    end
    local function compute_dist(originx,originy,destx,desty)
        return math.sqrt((originx-destx)*(originx-destx)+(originy-desty)*(originy-desty))
    end
    local function compute_len(pointdown1,pointdown2,pointup1,pointup2,borderlen)
       local dist1=3
       local dist2=3
       if judge_border(pointdown1[1],pointdown1[2],borderlen) and judge_border(pointdown2[1],pointdown2[2],borderlen) then
            dist1=compute_dist(pointdown1[1],pointdown1[2],pointdown2[1],pointdown2[2])
       end
       if judge_border(pointup1[1],pointup1[2],borderlen) and judge_border(pointup2[1],pointup2[2],borderlen) then
            dist2=compute_dist(pointup1[1],pointup1[2],pointup2[1],pointup2[2])
       end
       return math.floor((dist1+dist2)/4)
    end
    local mask=torch.zeros(64,64)
    for i = 1,nParts do
        mask=torch.cmax(mask,imgs[i])
    end--将多个分关节拼凑起来,从而融合到一张热图上面
    --image.save('show/'..imgname..'mymask'..imgname, mask)
    --绘制关节之间的连线,按顺序进行绘制
    for i = 1,nParts-1 do
        if judge_border(pts[i][1],pts[i][2],mask:size(1)) then
            if judge_border(pts[i+1][1],pts[i+1][2],mask:size(1)) then
                if i~=6 and i~=10  then--脚踝和骨盆，头上和手腕显然不能链接
                    if i==7 then--胸骨和骨盆 用于控制躯干宽度大小
                       len=compute_len(pts[3],pts[4],pts[13],pts[14],mask:size(1))--左右臀部，左右肩部
                    elseif i==2 or i==4 or i==1 or i==5 then--膝盖和臀部,膝盖到脚踝
                       len=2
                    elseif i==9 then
                       len=0
                    else--其他部位
                       len=1
                    end
                    mask=drawLine(mask,pts[i][1],pts[i][2],pts[i+1][1],pts[i+1][2],len)
                end
            end
        end
    end
    --补充一些其他关节点的连线
    --右臀和右肩膀
    if judge_border(pts[3][1],pts[3][2],mask:size(1)) then
       if judge_border(pts[13][1],pts[13][2],mask:size(1)) then
            len=1
            mask=drawLine(mask,pts[3][1],pts[3][2],pts[13][1],pts[13][2],len)
       end
    end
    --左臀和左肩膀
    if judge_border(pts[4][1],pts[4][2],mask:size(1)) then
       if judge_border(pts[14][1],pts[14][2],mask:size(1)) then
            len=1
            mask=drawLine(mask,pts[4][1],pts[4][2],pts[14][1],pts[14][2],len)
       end
    end
    --左臀和右肩膀
    if judge_border(pts[4][1],pts[4][2],mask:size(1)) then
       if judge_border(pts[13][1],pts[13][2],mask:size(1)) then
            len=1
            mask=drawLine(mask,pts[4][1],pts[4][2],pts[13][1],pts[13][2],len)
       end
    end
    --右臀和左肩膀
    if judge_border(pts[3][1],pts[3][2],mask:size(1)) then
       if judge_border(pts[14][1],pts[14][2],mask:size(1)) then
            len=1
            mask=drawLine(mask,pts[3][1],pts[3][2],pts[14][1],pts[14][2],len)
       end
    end
    return mask
end

function M.drawOffset(img, pt)

    -- img: 2xHxW offset field
    local h, w = img:size(2), img:size(3)
    assert(h == w)
    for i = 1, h do
        local dy = i - pt[2]
        img[{{2}, {i}, {}}] = dy
    end

    for j = 1, w do
        local dx = j - pt[1]
        img[{{1}, {}, {j}}] = dx
    end
    return img
end

-- -------------------------------------------------------------------------------
-- -- Draw offset map
-- -------------------------------------------------------------------------------
-- function M.drawOffsets(img, pt)
--     -- Img: x, y flow field 2xHxW
--     local h, w = img:size(2), img:size(3)
--     for i = 1, h do
--         for j = 1, w do
--             local img[1][i][j] = pt[1] - i
--             local img[2][i][j] = pt[2] - j
--         end
--     end
--     img[1] = img[1]/h
--     img[2] = img[2]/w
--     return img
-- end


-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------
function M.shuffleLR(x, dataset)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matchedParts
    if string.find(dataset, 'mpii') then
        matchedParts = {
            {1,6},   {2,5},   {3,4},
            {11,16}, {12,15}, {13,14}
        }
    elseif dataset == 'flic' or dataset == 'flic-sequence' then
        -- matchedParts = {
        --     {1,4}, {2,5}, {3,6}, {7,8}, {9,10}
        -- }
        
        matchedParts = {
          {1,2}, {4,8}, {5,9}, {6,10}, {7,11}
        }
    elseif dataset == 'lsp' then
        matchedParts = {
            {1,6}, {2,5}, {3,4}, {7,12}, {8,11}, {9,10}
        }
    elseif dataset == 'coco' then
        matchedParts = {
            {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {12,13},
            {14,15}, {16,17}
        }
    elseif dataset == 'cooking' then
        matchedParts = {
            {3,4}, {5,6}, {7,8}, {9,10}
        }
    elseif dataset == 'youtube' or dataset == 'youtube-single' 
        or string.find(dataset, 'bbcpose') ~= nil
        or string.find(dataset, 'videopose2') ~= nil then -- == 'bbcpose-single' or dataset == 'bbcpose' then        
        matchedParts = {
            {2,3}, {4,5}, {6,7}
        }
    else
      error('Not supported dataset: ' .. dataset)
    end

    for i = 1,#matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function M.flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end

function M.colorNormalize(img, meanstd)
  assert(img:size(1) == 3, ('images should be 3 channel (%d channel now)'):format(img:dim()))
  for i=1,3 do
     img[i]:add(-meanstd.mean[i])
     -- img[i]:div(meanstd.std[i])
  end
  return img
end


function M.colorNormalizeMeanImg(img, meanImg)
  assert(img:size(1) == 3, ('images should be 3 channel (%d channel now)'):format(img:dim()))
  assert(meanImg:size(1) == 3, ('meanImg should be 3 channel (%d channel now)'):format(meanImg:dim()))
  -- local mean = meanImg:mean(2):mean(3):squeeze()
  -- for i=1,3 do
  --    img[i]:add(-mean[i])
  -- end
  -- img = img - meanImg
  img:csub(meanImg)
  return img
end


local function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
   dst:resizeAs(img)
   dst[1]:zero()
   dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
   dst[2]:copy(dst[1])
   dst[3]:copy(dst[1])
   return dst
end

function M.Saturation(input, var)
      if var == 0 then return input end
      local gs = gs or input.new()
      grayscale(gs, input)

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
end

function M.Brightness(input, var)
      if var == 0 then return input end
      local gs
      gs = gs or input.new()
      gs:resizeAs(input):zero()

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
end

function M.Contrast(input, var)
      if var == 0 then return input end
      local gs
      gs = gs or input.new()
      grayscale(gs, input)
      gs:fill(gs[1]:mean())

      local alpha = 1.0 + torch.uniform(-var, var)
      blend(input, gs, alpha)
      return input
end

function M.colorJitter(input, brightness, contrast, saturation)
   local brightness = brightness or 0
   local contrast = contrast or 0
   local saturation = saturation or 0

   local ts = {'Brightness', 'Contrast', 'Saturation'}
   local var = {brightness, contrast, saturation}
   local order = torch.randperm(#ts)
   for i=1,#ts do
      input = M[ts[order[i]]](input, var[order[i]])
   end

   return input
end

function M.colorNoise(input, var)

   if var == 0 then return input end
   local h, w = input:size(2), input:size(3)
   local gs = torch.Tensor(1, h, w):normal(0, 0.2)
   local mask = torch.Tensor(1, h, w):uniform(0, 1)

   gs[mask:gt(var)] = 0

   input = input + gs:expandAs(input)
   return input:clamp(0, 1)
end

function M.gaussianBlur(input, var)
   local kw = math.floor(torch.uniform(0, var))
   if torch.uniform() <= .6 then kw = 0 end

   if kw ~= 0 then
      local k = image.gaussian{width=3, normalize=true}:typeAs(input)
      return image.convolve(input, k, 'same'):contiguous()
   end
   return input
end

return M