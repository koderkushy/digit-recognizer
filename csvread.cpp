#include<bits/stdc++.h>
using namespace std;
#define ll long long 
#define pb push_back
#define ub upper_bound
#define lb lower_bound
#define mp make_pair
#define umap unordered_map
#define popcount(x) __builtin_popcountll(x)
#define all(v) v.begin() , v.end()
#define PI 3.141592653589793238
#define E 2.7182818284590452353602874713527
#define M 1000000007
const long long INF = 1e18;
int main()
{
ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
//freopen("input.txt", "r", stdin);
//freopen("output.txt", "w", stdout);
string s;
cin>>s;
 
vector<vector<string>> data;
vector<string> row;
string line, number;
 
fstream file (s, ios::in);
if(file.is_open())
{
while(getline(file, line))
{
row.clear();
 
stringstream str(line);
 
while(getline(str, number, ','))
row.push_back(number);

data.push_back(row);
}
}
// for(int i=0;i<data.size();i++)
// {
//     for(int j=0;j<data[i].size();j++)
//     {
//         cout<<data[i][j]<<" ";
//     }
//     cout<<endl;
// }

vector<vector<double>>value(data.size(),vector<double>(data[0].size()));
for(int i=0;i<data.size();i++)
{
    for(int j=0;j<data[i].size();j++)
    {
    value[i][j]=stod(data[i][j]);
    }
    cout<<"\n";
}
// double xx=stod(data[0][0]);
for(int i=0;i<value.size();i++)
{
    for(int j=0;j<value[i].size();j++)
    {
        cout<<value[i][j]<<" ";
    }
    cout<<"\n";
}
 

}