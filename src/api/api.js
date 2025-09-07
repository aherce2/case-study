import { BACKEND_URL } from "./config"; 

export const getAIMessage = async (userQuery) => {
  const response = await fetch(`${BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ question: userQuery }),
  });
  const data = await response.json();
  return {
    role: "assistant",
    content: data.response,
  };
};


// export const getAIMessage = async (userQuery) => {

//   const message = 
//     {
//       role: "assistant",
//       content: "Connect your backend here...."
//     }

//   return message;
// };