local dkjson = require( "game/dkjson" )

local request = {}

function request:Log(item)
  request = CreateHTTPRequest(":8080")
  request:SetHTTPRequestRawPostBody("application/json", dkjson.encode(GetCursorLocation()))
  -- request:Send(function(response) end)
end

return request;
