local dkjson = require( "game/dkjson" )
local utils = require(GetScriptDirectory() .. "/utils")

local request = {}

function request:Send(data, callback)
  local retVal
  request = CreateHTTPRequest(":8080")
  request:SetHTTPRequestRawPostBody('application/json', dkjson.encode(data))
  request:Send(function(res)
    if res.StatusCode ~= 200 then
      print("POST failed")
      return
    end
    callback(dkjson.decode(res.Body))
   end)
end

return request;
