dkjson = require( "game/dkjson" )

function Log(item)
  request = CreateHTTPRequest(":5000")
  request:SetHTTPRequestRawPostBody('application/json', dkjson.encode(GetCursorLocation()))
  request:Send(function(response) end)
end

return Log;
